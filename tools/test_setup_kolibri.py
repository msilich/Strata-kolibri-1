"""Kolibri's source-only setup: mocked downloads, compiler, tokenizer and server.

    python -m unittest tools.test_setup_kolibri
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup
from test_setup_golden import install as main_install


GPU = {"index": 0, "count": 1, "name": "Test CUDA GPU", "arch": "120", "vram_gb": 32.0,
       "driver": "580.97"}
FILENAME = "Kolibri-1-Q4_K_M.gguf"
FAST_FLAGS = {
    "--split-k-attention", "--split-k-local-attention", "--grouped-experts", "--decode-cache-lru",
    "--prefill-attention", "--fused-decode", "--tensor-decode-attention", "--grouped-prefill",
    "--gpu-greedy", "--pin-experts", "--overlap-prefill-uploads", "--gpu-expert-cache", "--kv-grow",
    "--overlap-decode-cache", "--two-query-prefill",
}


def args(**kw):
    values = dict(family="kolibri", model="Q4_K_M", yes=True, vision=None, context=None, parallel=None,
                  gpu=None, gpus=None, layer_split=None, gpu_roles=None, models_dir=None, gguf_dir=None,
                  data_dir=None, host=None, api_key=None, browser=None, no_start=True, check=False,
                  backend="cuda", cuda=None, rope_scaling=None, rope_scale=None, kv=None,
                  resident_budget_gib=None, low_ram="auto", vram_reserve_mib=None)
    values.update(kw)
    return argparse.Namespace(**values)


def quiet(fn, *a, **kw):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            return fn(*a, **kw), output.getvalue()
        except SystemExit as exc:
            return exc.code, output.getvalue()


class KolibriEntries(unittest.TestCase):
    def test_source_family_uses_the_existing_download_pin(self):
        family = setup.SOURCE_FAMILIES["kolibri"]
        self.assertEqual(family["arch"], "kolibri1")
        self.assertEqual(list(family["models"]), ["Q4_K_M"])
        self.assertEqual(family["file"].format(q="Q4_K_M"), FILENAME)
        self.assertRegex(family["hf"], r"/resolve/[0-9a-f]{40}/$")
        self.assertEqual(setup.file_pin(family["hf"] + FILENAME), family["sha256"][FILENAME])
        size, sha = family["sha256"][FILENAME]
        self.assertGreater(size, 40_000_000_000)
        self.assertEqual(sha, "2f2670a78b5c21bef65ef286791285d8e0ad79aef5f0904a3784cb7eeadd5f3e")
        for families in (setup.FAMILIES, setup.OWN_PATH_FAMILIES, setup.MULTI_FILE_FAMILIES):
            self.assertNotIn("kolibri", families)

    def test_menu_only_offers_source_cuda_linux(self):
        for platform, win, control, hip, expected in (
            ("linux", False, False, False, ["kolibri"]),
            ("linux", False, False, True, []),
            ("linux", False, True, False, []),
            ("win32", True, False, False, []),
            ("darwin", False, False, False, []),
        ):
            with self.subTest(platform=platform, control=control, hip=hip), \
                    mock.patch.object(sys, "platform", platform), mock.patch.object(setup, "WIN", win), \
                    mock.patch.object(setup, "CONTROL", control):
                self.assertEqual(setup.source_menu_families(hip), expected)

    def test_cli_and_interactive_menu_dispatch_to_the_isolated_installer(self):
        with mock.patch.object(sys, "platform", "linux"), mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "CONTROL", False), mock.patch.object(setup, "OLD_GPUS", None):
            families = (setup.menu_families(False) + list(setup.OWN_PATH_FAMILIES)
                        + list(setup.MULTI_FILE_FAMILIES) + setup.source_menu_families(False))
            for argv, answers in ((["--family", "kolibri", "--model", "Q4_K_M", "--no-start"], None),
                                  (["--no-start"],
                                   {"Which model?": str(families.index("kolibri") + 1)})):
                with self.subTest(argv=argv):
                    installer = mock.Mock(return_value=0)
                    code, output, _, _ = main_install(64.0, [dict(GPU)], argv, answers=answers,
                        extra=[mock.patch.object(setup, "install_kolibri", installer)])
                    self.assertEqual(code, 0, output)
                    installer.assert_called_once()
                    self.assertEqual(installer.call_args.args[0].model, "Q4_K_M" if answers is None else None)
                    if answers is not None:
                        self.assertIn("[experimental source]", output)

    def test_explicit_family_is_rejected_before_other_backend_setup(self):
        for platform, backend in (("darwin", []), ("linux", ["--backend", "sycl"])):
            with self.subTest(platform=platform), mock.patch.object(sys, "platform", platform), \
                    mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "CONTROL", False), \
                    mock.patch.object(sys, "argv", ["setup.py", "--family", "kolibri", "--model", "Q4_K_M",
                                                   "--yes", "--no-start", *backend]), \
                    mock.patch.object(setup, "mac_setup", side_effect=AssertionError("delegated to Metal")), \
                    mock.patch.object(setup, "sycl_setup", side_effect=AssertionError("delegated to SYCL")), \
                    mock.patch.object(setup, "download", side_effect=AssertionError("download before rejection")), \
                    mock.patch.object(setup, "pip_install", side_effect=AssertionError("install before rejection")):
                code, output = quiet(setup.main)
                self.assertEqual(code, 1, output)
                self.assertIn("Kolibri", output)


class KolibriInstall(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.eng = self.root / "engine-kolibri"
        self.eng.mkdir()
        (self.eng / "strata-kolibri").write_bytes(b"")
        (self.eng / "BUILD.json").write_text(json.dumps({"archs": [120], "cuda_dirs": ["<cuda>"]}))
        self.cfg = {}
        self.cfg_path = None
        self.start_impl = setup.start

        def write_config(path, cfg, source=None, explicit=None):
            self.cfg, self.cfg_path = cfg, path
            path.write_text(json.dumps(cfg), encoding="utf-8")

        def download(url, path, *a, **kw):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"GGUF")

        patches = {
            "ROOT": mock.patch.object(setup, "ROOT", self.root),
            "WIN": mock.patch.object(setup, "WIN", False),
            "CONTROL": mock.patch.object(setup, "CONTROL", False),
            "platform": mock.patch.object(sys, "platform", "linux"),
            "download": mock.patch.object(setup, "download", side_effect=download),
            "verify": mock.patch.object(setup, "verify_sha256"),
            "disk": mock.patch.object(setup, "free_gb", return_value=500.0),
            "pip": mock.patch.object(setup, "pip_install"),
            "llama": mock.patch.object(setup, "get_llama_cpp", return_value=self.root / "llama.cpp"),
            "build": mock.patch.object(setup, "build_kolibri_engine", return_value=self.eng),
            "run": mock.patch.object(setup, "run"),
            "data": mock.patch.object(setup, "data_folder_for", return_value=(self.root / "data", [])),
            "config": mock.patch.object(setup, "write_setup_config", side_effect=write_config),
            "start": mock.patch.object(setup, "start", return_value=0),
        }
        self.mocks = {}
        for name, patch in patches.items():
            self.mocks[name] = patch.start()
            self.addCleanup(patch.stop)

    def install(self, ram=64.0, hip=False, gpu=None, **kw):
        options = args(models_dir=str(self.root / "models"), **kw)
        return quiet(setup.install_kolibri, options, gpu or dict(GPU), hip, 8080, ram, None)

    def test_pinned_download_tokenizer_config_and_real_run_script(self):
        code, output = self.install()
        self.assertEqual(code, 0, output)
        family = setup.SOURCE_FAMILIES["kolibri"]
        gguf = Path(self.cfg["args"][self.cfg["args"].index("--native") + 1])
        self.assertEqual(gguf.name, FILENAME)
        self.assertEqual(self.mocks["download"].call_args.args[:2], (family["hf"] + FILENAME, gguf))
        self.mocks["verify"].assert_called_once_with(gguf, *family["sha256"][FILENAME], own=False)
        self.mocks["build"].assert_called_once_with(GPU, True)
        self.mocks["pip"].assert_called_once()
        self.mocks["run"].assert_called_once()
        command = self.mocks["run"].call_args.args[0]
        self.assertEqual(Path(command[1]).name, "strata_tokenizer.py")
        self.assertEqual(command[command.index("--gguf") + 1], str(gguf))
        self.assertEqual(Path(command[command.index("--out") + 1]) / "tokenizer", Path(self.cfg["tokenizer"]))
        self.assertEqual(self.cfg["exe"], str(self.eng / "strata-kolibri"))
        self.assertEqual(self.cfg["arch"], "kolibri1")
        self.assertEqual(self.cfg["parallel"], 1)
        self.assertEqual(self.cfg["host"], "127.0.0.1")
        self.assertEqual(self.cfg["lib_dirs"], ["<cuda>"])
        engine_args = self.cfg["args"]
        for flag, value in (("--max-context", "32768"), ("--chunk", "4096"), ("--expert-vram", "-1")):
            self.assertEqual(engine_args[engine_args.index(flag) + 1], value)
        self.assertTrue(FAST_FLAGS.issubset(engine_args))
        self.assertFalse({"--pack", "--mtp", "--vision", "--kv"}.intersection(engine_args))
        self.assertNotIn("vision", self.cfg)
        scripts = list(self.root.glob("run-kolibri-*.sh"))
        self.assertEqual(len(scripts), 1)
        script = scripts[0].read_text()
        self.assertIn(str(self.cfg_path), script)
        self.assertIn('"--port" "8080"', script)
        self.assertTrue(scripts[0].stat().st_mode & 0o111)
        self.mocks["start"].assert_not_called()

    def test_own_file_is_verified_without_download_or_disk_requirement(self):
        folder = self.root / "own"
        folder.mkdir()
        gguf = folder / FILENAME
        gguf.write_bytes(b"GGUF")
        self.mocks["disk"].return_value = 0.0
        code, output = self.install(gguf_dir=str(folder))
        self.assertEqual(code, 0, output)
        self.mocks["download"].assert_not_called()
        self.mocks["verify"].assert_called_once_with(
            gguf, *setup.SOURCE_FAMILIES["kolibri"]["sha256"][FILENAME], own=True)

    def test_missing_own_file_and_insufficient_disk_do_not_download(self):
        for options in ({"gguf_dir": str(self.root / "missing")}, {}):
            with self.subTest(options=options):
                self.mocks["disk"].return_value = 0.0
                code, output = self.install(**options)
                self.assertEqual(code, 1, output)
                self.mocks["download"].assert_not_called()
                self.mocks["config"].assert_not_called()

    def test_resumed_download_needs_only_the_missing_disk_bytes(self):
        folder = self.root / "models" / "Kolibri-1"
        folder.mkdir(parents=True)
        (folder / (FILENAME + ".part")).write_bytes(b"x" * 800)
        self.mocks["disk"].return_value = 300 / 1e9
        pins = setup.SOURCE_FAMILIES["kolibri"]["sha256"]
        with mock.patch.dict(pins, {FILENAME: (1000, pins[FILENAME][1])}):
            code, output = self.install()
        self.assertEqual(code, 0, output)
        self.mocks["download"].assert_called_once()

    def test_sha_failure_does_not_export_tokenizer_or_write_config(self):
        self.mocks["verify"].side_effect = SystemExit(1)
        code, output = self.install()
        self.assertEqual(code, 1, output)
        self.mocks["run"].assert_not_called()
        self.mocks["config"].assert_not_called()
        self.mocks["start"].assert_not_called()

    def test_unsupported_platform_or_gpu_fails_before_acquisition(self):
        for platform, win, control, hip, arch in (
            ("win32", True, False, False, "120"), ("darwin", False, False, False, "120"),
            ("linux", False, True, False, "120"), ("linux", False, False, True, "120"),
            ("linux", False, False, False, "75"),
        ):
            with self.subTest(platform=platform, control=control, hip=hip, arch=arch), \
                    mock.patch.object(sys, "platform", platform), mock.patch.object(setup, "WIN", win), \
                    mock.patch.object(setup, "CONTROL", control):
                code, output = self.install(hip=hip, gpu={**GPU, "arch": arch})
                self.assertEqual(code, 1, output)
                for name in ("download", "pip", "llama", "build", "config"):
                    self.mocks[name].assert_not_called()

    def test_unsupported_options_fail_before_acquisition(self):
        for options in ({"model": "Q8_0"}, {"vision": "yes"}, {"vision": "cpu"}, {"vision": "gpu"},
                        {"parallel": 2}, {"gpus": "0,1"}, {"layer_split": "25"}, {"context": 262145}):
            with self.subTest(options=options):
                code, output = self.install(**options)
                self.assertEqual(code, 1, output)
                for name in ("download", "pip", "llama", "build", "config"):
                    self.mocks[name].assert_not_called()

    def test_low_ram_uses_existing_risk_confirmation(self):
        with mock.patch.object(setup, "confirm_risk", side_effect=SystemExit(1)) as confirm:
            code, output = self.install(ram=32.0)
        self.assertEqual(code, 1, output)
        confirm.assert_called_once()
        self.mocks["download"].assert_not_called()

    def test_explicit_context_start_and_config_read_back(self):
        code, output = self.install(context=262144, vision="no", no_start=False)
        self.assertEqual(code, 0, output)
        self.mocks["start"].assert_called_once()
        options = setup.choices_from_config(self.cfg_path)
        self.assertEqual((options["family"], options["model"], options["context"]),
                         ("kolibri", "Q4_K_M", 262144))
        self.assertEqual(setup.calibration_targets([self.cfg_path], None, None), [])
        self.assertEqual(setup.calibration_targets([self.cfg_path], "Q4_K_M", "kolibri"), [])

    def test_restart_uses_saved_source_engine_without_installing(self):
        code, output = self.install()
        self.assertEqual(code, 0, output)
        saved = self.cfg_path.read_bytes()
        for name in ("download", "pip", "llama", "build"):
            self.mocks[name].reset_mock()
        with mock.patch.object(setup, "gpus", return_value=[dict(GPU)]), \
                mock.patch.object(setup, "OLD_GPUS", None), \
                mock.patch.object(setup, "old_gpus_opt_in", return_value=None), \
                mock.patch.object(setup, "engine_runs_on", side_effect=AssertionError("generic engine lookup")), \
                mock.patch.object(setup, "build_engine", side_effect=AssertionError("generic engine build")), \
                mock.patch.dict(os.environ, {"STRATA_EXECV": ""}), \
                mock.patch.object(setup.subprocess, "call", return_value=0) as server:
            code, output = quiet(self.start_impl, self.cfg_path, None, open_browser=False, yes=True)
        self.assertEqual(code, 0, output)
        server.assert_called_once()
        command = server.call_args.args[0]
        self.assertEqual(command[command.index("--config") + 1], str(self.cfg_path))
        self.assertEqual(server.call_args.kwargs["env"][setup.acquisition.ENV], "1")
        self.assertEqual(self.cfg_path.read_bytes(), saved)
        for name in ("download", "pip", "llama", "build"):
            self.mocks[name].assert_not_called()

    def test_restart_incompatible_architecture_requires_explicit_repair(self):
        code, output = self.install()
        self.assertEqual(code, 0, output)
        self.mocks["build"].reset_mock()
        with mock.patch.object(setup, "build_engine", side_effect=AssertionError("generic engine build")), \
                mock.patch.object(setup, "engine_runs_on", side_effect=AssertionError("generic engine lookup")):
            code, output = quiet(setup.ensure_engine_for, [{**GPU, "arch": "86"}], self.cfg_path, self.cfg, True)
        self.assertEqual(code, 1, output)
        self.assertIn("--setup", output)
        self.mocks["build"].assert_not_called()

    def test_unsupported_start_overrides_leave_the_profile_unchanged(self):
        code, output = self.install()
        self.assertEqual(code, 0, output)
        saved = self.cfg_path.read_bytes()
        for options in ({"gpu": [0, 1]}, {"layer_split": "25"}, {"keep": {"vram_reserve_mib": 2048}}):
            with self.subTest(options=options):
                code, output = quiet(self.start_impl, self.cfg_path, None, **options)
                self.assertEqual(code, 1, output)
                self.assertEqual(self.cfg_path.read_bytes(), saved)

    def test_update_only_builds_the_kolibri_source_engine(self):
        code, output = self.install()
        self.assertEqual(code, 0, output)
        self.mocks["build"].reset_mock()
        options = args(build=False, prebuilt=None)
        with mock.patch.object(setup, "gpu_info", return_value=dict(GPU)), \
                mock.patch.object(setup, "update_installed_engine", side_effect=AssertionError("release update")), \
                mock.patch.object(setup, "refresh_shared_draft_vocab", side_effect=AssertionError("MTP update")), \
                mock.patch.object(setup, "engine_version", return_value=setup.MIN_ENGINE):
            code, output = quiet(setup.update_install, [self.cfg_path], options)
        self.assertEqual(code, 0, output)
        self.mocks["build"].assert_called_once_with(GPU, True)
        updated = json.loads(self.cfg_path.read_text())
        self.assertEqual(updated["exe"], str(self.eng / "strata-kolibri"))
        self.assertEqual(updated["args"], self.cfg["args"])


class KolibriBuild(unittest.TestCase):
    def test_cuda_target_and_libraries_are_isolated_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            old_engine = root / "engine" / setup.EXE
            old_engine.parent.mkdir()
            old_engine.write_bytes(b"existing release engine")
            nvcc = root / "cuda" / "bin" / "nvcc"
            nvcc.parent.mkdir(parents=True)
            (root / "cuda" / "lib64").mkdir()

            def compile_target(src, bdir, target, defs, vcvars, bat_name):
                bdir.mkdir(parents=True, exist_ok=True)
                (bdir / target).write_bytes(b"Kolibri engine")
                (bdir / setup.KQ_LIB).write_bytes(b"matching KQ library")

            for patch in (mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "CONTROL", False),
                          mock.patch.object(setup, "WIN", False), mock.patch.object(sys, "platform", "linux"),
                          mock.patch.object(setup, "source_hash", return_value="test-source"),
                          mock.patch.object(setup, "source_version", return_value="test-version"),
                          mock.patch.object(setup, "get_llama_cpp", return_value=root / "llama.cpp"),
                          mock.patch.object(setup, "install_build_tools", return_value=(str(nvcc), None)),
                          mock.patch.object(setup, "host_compiler_defs", return_value=[]),
                          mock.patch.object(setup.acquisition, "require")):
                stack.enter_context(patch)
            compiler = stack.enter_context(mock.patch.object(setup, "cmake_build", side_effect=compile_target))
            engine, _ = quiet(setup.build_kolibri_engine, dict(GPU), True)
            compiler.assert_called_once()
            src, bdir, target, defs, _, _ = compiler.call_args.args
            self.assertEqual(src, root)
            self.assertEqual(bdir, root / "build-kolibri")
            self.assertEqual(target, "strata-kolibri")
            for definition in ("-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_ENABLE_HIP=OFF",
                               "-DSTRATA_ENABLE_SYCL=OFF", "-DSTRATA_BUILD_KOLIBRI=ON",
                               "-DCMAKE_CUDA_ARCHITECTURES=120"):
                self.assertIn(definition, defs)
            self.assertEqual(engine, root / "engine-kolibri")
            self.assertEqual((engine / "strata-kolibri").read_bytes(), b"Kolibri engine")
            self.assertEqual((engine / setup.KQ_LIB).read_bytes(), b"matching KQ library")
            stamp = json.loads((engine / "BUILD.json").read_text())
            self.assertEqual(stamp["archs"], [120])
            self.assertIn(str(root / "cuda" / "lib64"), stamp["cuda_dirs"])
            again, _ = quiet(setup.build_kolibri_engine, dict(GPU), True)
            self.assertEqual(again, engine)
            compiler.assert_called_once()
            self.assertEqual(old_engine.read_bytes(), b"existing release engine")


if __name__ == "__main__":
    unittest.main()
