import unittest
from pathlib import Path
import shutil
import subprocess
import tempfile

from mtc_compiler.codegen import compile_model, emit_project
from mtc_compiler.config import ConfigError, load_model, model_from_dict
from mtc_compiler.analysis import analyze_ranges
from mtc_compiler.ir import build_stages
from mtc_compiler.memory_planner import plan_memory
from mtc_compiler.range_analysis import Interval
from mtc_compiler.validate import validate_model


EXAMPLE = Path(__file__).parents[1] / "examples" / "xor_net.json"
SENSOR_EXAMPLE = Path(__file__).parents[1] / "examples" / "sensor_drift_net.json"


class ModelValidationTests(unittest.TestCase):
    def test_example_model_is_valid(self):
        model = load_model(EXAMPLE)
        validate_model(model)
        self.assertEqual(model.name, "xor_net")

    def test_builds_dense_stages_with_activations(self):
        stages = build_stages(load_model(EXAMPLE))
        self.assertEqual(
            [(stage.in_features, stage.out_features, stage.activation) for stage in stages],
            [(2, 4, "relu"), (4, 1, "sigmoid")],
        )

    def test_range_analysis_marks_possible_q15_overflow(self):
        model = load_model(EXAMPLE)
        report = analyze_ranges(model, build_stages(model))
        self.assertGreater(len(report.overflows), 0)
        self.assertGreater(report.stability_score, 1.0)

    def test_memory_plan_is_static_and_within_demo_budget(self):
        model = load_model(EXAMPLE)
        stages = build_stages(model)
        plan = plan_memory(model, stages)
        self.assertLessEqual(plan.total_static_ram_bytes, plan.budget_bytes)
        self.assertEqual(plan.pool_elements, max(item.end_elements for item in plan.allocations))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            plan_memory(model, stages, budget_bytes=1)

    def test_compile_requires_explicit_clipping_for_predicted_overflow(self):
        model = load_model(EXAMPLE)
        with self.assertRaisesRegex(ConfigError, "Q15 overflow predicted"):
            compile_model(model)

    def test_wokwi_target_emits_arduino_project_files(self):
        compilation = compile_model(load_model(EXAMPLE), target="wokwi", allow_clipping=True)
        self.assertEqual(
            set(compilation.files),
            {"mtc_model.h", "mtc_model.c", "sketch.ino", "diagram.json"},
        )
        self.assertIn('extern "C"', compilation.files["mtc_model.h"])
        self.assertIn("Serial.begin(115200)", compilation.files["sketch.ino"])
        self.assertIn('Serial.printf("MTC ESP32 trainer: xor_net\\n")', compilation.files["sketch.ino"])
        self.assertNotIn('trainer: "xor_net"', compilation.files["sketch.ino"])
        self.assertIn('Serial.printf("sample=%u target=', compilation.files["sketch.ino"])
        self.assertIn("mtc_train_sample", compilation.files["sketch.ino"])

    def test_rejects_disconnected_dense_layers(self):
        model = model_from_dict({
            "name": "broken",
            "precision": "int16",
            "layers": [
                {"type": "dense", "in_features": 2, "out_features": 4},
                {"type": "dense", "in_features": 3, "out_features": 1},
            ],
            "loss": "mse",
            "optimizer": {"type": "sgd", "learning_rate": 0.1},
        })
        with self.assertRaisesRegex(ConfigError, "Layer 1: Input features"):
            validate_model(model)

    def test_rejects_activation_as_first_layer(self):
        model = model_from_dict({
            "name": "broken",
            "precision": "int16",
            "layers": [{"type": "relu"}],
            "loss": "mse",
            "optimizer": {"type": "sgd", "learning_rate": 0.1},
        })
        with self.assertRaisesRegex(ConfigError, "cannot be used as the first layer"):
            validate_model(model)

    def test_rejects_consecutive_activations(self):
        model = model_from_dict({
            "name": "broken",
            "precision": "int16",
            "layers": [
                {"type": "dense", "in_features": 1, "out_features": 1},
                {"type": "relu"},
                {"type": "sigmoid"},
            ],
            "loss": "mse",
            "optimizer": {"type": "sgd", "learning_rate": 0.1},
        })
        with self.assertRaisesRegex(ConfigError, "Consecutive activation"):
            validate_model(model)

    def test_rejects_training_sample_with_wrong_shape(self):
        model = model_from_dict({
            "name": "broken",
            "precision": "int16",
            "layers": [
                {"type": "dense", "in_features": 2, "out_features": 1},
                {"type": "sigmoid"},
            ],
            "target_range": [0.0, 1.0],
            "loss": "mse",
            "optimizer": {"type": "sgd", "learning_rate": 0.1},
            "training_data": [{"input": [0.0], "target": [1.0]}],
        })
        with self.assertRaisesRegex(ConfigError, "expected 2 inputs, got 1"):
            validate_model(model)

    def test_rejects_unknown_model_fields(self):
        with self.assertRaisesRegex(ConfigError, "unsupported field.*layres"):
            model_from_dict({
                "name": "broken",
                "precision": "int16",
                "layres": [],
                "layers": [{"type": "dense", "in_features": 1, "out_features": 1}],
                "loss": "mse",
                "optimizer": {"type": "sgd", "learning_rate": 0.1},
            })

    def test_rejects_learning_rate_that_quantizes_to_zero(self):
        with self.assertRaisesRegex(ConfigError, "rounds to zero in Q15"):
            model_from_dict({
                "name": "broken",
                "precision": "int16",
                "layers": [{"type": "dense", "in_features": 1, "out_features": 1}],
                "loss": "mse",
                "optimizer": {"type": "sgd", "learning_rate": 0.000001},
            })

    def test_rejects_negative_targets_for_sigmoid_output(self):
        model = model_from_dict({
            "name": "broken",
            "precision": "int16",
            "layers": [
                {"type": "dense", "in_features": 1, "out_features": 1},
                {"type": "sigmoid"},
            ],
            "target_range": [-1.0, 1.0],
            "loss": "mse",
            "optimizer": {"type": "sgd", "learning_rate": 0.1},
        })
        with self.assertRaisesRegex(ConfigError, "target_range minimum"):
            validate_model(model)


class IntervalTests(unittest.TestCase):
    def test_q15_overflow_and_relu(self):
        interval = Interval(-2.0, 0.5)
        self.assertTrue(interval.overflows_q15)
        self.assertEqual(interval.relu(), Interval(0.0, 0.5))

    def test_clamp_q15_handles_intervals_outside_range(self):
        self.assertEqual(Interval(2.0, 3.0).clamp_q15(), Interval(1.0 - 1 / 32768, 1.0 - 1 / 32768))
        self.assertEqual(Interval(-3.0, -2.0).clamp_q15(), Interval(-1.0, -1.0))

    def test_sigmoid_range(self):
        result = Interval(-5.0, 5.0).sigmoid()
        self.assertGreater(result.min, 0.0)
        self.assertLess(result.min, 0.01)
        self.assertGreater(result.max, 0.99)
        self.assertLess(result.max, 1.0)

    def test_dot_product_checks_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            Interval.dot_product([Interval(0.0, 1.0)], [])


@unittest.skipUnless(shutil.which("gcc"), "GCC is required for generated-C integration")
class GeneratedCIntegrationTests(unittest.TestCase):
    def test_generated_training_programs_compile_run_and_learn(self):
        for model_path in (EXAMPLE, SENSOR_EXAMPLE):
            with self.subTest(model=model_path.stem):
                model = load_model(model_path)
                with tempfile.TemporaryDirectory() as directory:
                    project = Path(directory)
                    emit_project(model, project, allow_clipping=True)
                    model_source = (project / "mtc_model.c").read_text(encoding="utf-8")
                    self.assertIn("static int16_t mtc_memory_pool", model_source)
                    self.assertNotIn("static int16_t mtc_weights[", model_source)
                    self.assertNotIn("malloc(", model_source)
                    cmake = shutil.which("cmake")
                    if cmake:
                        build = project / "build"
                        subprocess.run(
                            [cmake, "-S", str(project), "-B", str(build)],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        subprocess.run(
                            [cmake, "--build", str(build), "--parallel", "2"],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        executable = build / "mtc_train"
                    else:
                        executable = project / "mtc_train"
                        subprocess.run(
                            [
                                "gcc", "-std=c99", "-Wall", "-Wextra", "-Werror",
                                "mtc_model.c", "mtc_train.c", "-o", str(executable),
                            ],
                            cwd=project,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    result = subprocess.run(
                        [str(executable)], check=True, capture_output=True, text=True
                    )
                    lines = result.stdout.splitlines()
                    self.assertEqual(len(lines), model.epochs)
                    self.assertTrue(lines[0].startswith("epoch=1 loss_q15="))
                    self.assertTrue(lines[-1].startswith(f"epoch={model.epochs} loss_q15="))
                    first_loss = int(lines[0].split("loss_q15=")[1])
                    final_loss = int(lines[-1].split("loss_q15=")[1])
                    self.assertLess(final_loss, first_loss)

    @unittest.skipUnless(shutil.which("gcc"), "GCC is required for C99 validation")
    def test_wokwi_model_core_compiles_as_c99(self):
        compilation = compile_model(
            load_model(EXAMPLE), target="wokwi", allow_clipping=True
        )
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for filename in ("mtc_model.h", "mtc_model.c"):
                (project / filename).write_text(compilation.files[filename], encoding="utf-8")
            subprocess.run(
                ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-c", "mtc_model.c"],
                cwd=project,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_xor_example_classifies_all_samples_after_training(self):
        model = load_model(EXAMPLE)
        compilation = compile_model(model, target="host", allow_clipping=True)
        to_q15 = lambda value: max(-32768, min(32767, round(value * 32768)))
        input_rows = ",\n".join(
            "{" + ", ".join(str(to_q15(value)) for value in sample.inputs) + "}"
            for sample in model.training_data
        )
        target_rows = ", ".join(
            str(to_q15(sample.targets[0])) for sample in model.training_data
        )
        harness = f'''#include "mtc_model.h"
#include <stdint.h>
#include <stdio.h>
static const int16_t inputs[MTC_TRAIN_SAMPLES][MTC_INPUT_SIZE] = {{
{input_rows}
}};
static const int16_t targets[MTC_TRAIN_SAMPLES] = {{{target_rows}}};
int main(void) {{
    mtc_reset();
    for (uint32_t epoch = 0; epoch < MTC_EPOCHS; ++epoch) {{
        for (uint32_t sample = 0; sample < MTC_TRAIN_SAMPLES; ++sample) {{
            (void)mtc_train_sample(inputs[sample], &targets[sample]);
        }}
    }}
    uint32_t correct = 0;
    int16_t prediction[MTC_OUTPUT_SIZE];
    for (uint32_t sample = 0; sample < MTC_TRAIN_SAMPLES; ++sample) {{
        mtc_forward(inputs[sample], prediction);
        if ((prediction[0] >= 16384) == (targets[sample] >= 16384)) ++correct;
    }}
    printf("%u/%u\\n", (unsigned)correct, (unsigned)MTC_TRAIN_SAMPLES);
    return 0;
}}
'''
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "mtc_model.h").write_text(compilation.files["mtc_model.h"], encoding="utf-8")
            (project / "mtc_model.c").write_text(compilation.files["mtc_model.c"], encoding="utf-8")
            (project / "check.c").write_text(harness, encoding="utf-8")
            executable = project / "check"
            build = subprocess.run(
                ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "mtc_model.c", "check.c", "-o", str(executable)],
                cwd=project,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run(
                [str(executable)], check=True, capture_output=True, text=True
            )
            self.assertEqual(result.stdout.strip(), "4/4")


if __name__ == "__main__":
    unittest.main()