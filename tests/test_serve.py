import json
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

from env.scenario import load_scenario
from viz.serve import VIZ_DIR, Handler, live_scenario


def test_live_scenario_scales_copy_only():
    before = load_scenario("s4_turbulent")
    s = live_scenario("s4_turbulent", 0.5, 2.0)
    assert s["wind"][0]["speed_mm_s"] == before["wind"][0]["speed_mm_s"] * 0.5
    assert s["wind"][1]["sigma_mm_s"] == before["wind"][1]["sigma_mm_s"] * 0.5
    assert s["wind"][1]["tau_c_s"] == before["wind"][1]["tau_c_s"]
    assert s["noise_multiplier"] == 2.0
    assert s["name"] == "s4_turbulent_w0.5_n2"
    assert load_scenario("s4_turbulent") == before


def test_live_scenario_default_keeps_name():
    assert live_scenario("s3_gust", 1.0, 1.0) == load_scenario("s3_gust")


@pytest.mark.parametrize("wind, noise", [(2.5, None), (-1.0, None), (1.0, 5.0),
                                         (float("nan"), None)])
def test_live_scenario_rejects_out_of_range(wind, noise):
    with pytest.raises(ValueError):
        live_scenario("s2_crosswind", wind, noise)


def test_live_scenario_respects_wind_caps():
    s2 = load_scenario("s2_crosswind")
    limit = 150.0 / s2["wind"][0]["speed_mm_s"]
    if limit < 2.0:
        with pytest.raises(ValueError):
            live_scenario("s2_crosswind", min(2.0, limit + 0.1), None)
    assert live_scenario("s2_crosswind", min(limit, 2.0), None)


@pytest.fixture
def server(tmp_path):
    handler = type("TestHandler", (Handler,), {"runs_dir": tmp_path})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(VIZ_DIR)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    httpd.server_close()


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_live_stream_with_sliders(server):
    base, runs = server
    status, body = get(f"{base}/api/live?scenario=s2_crosswind&controller=straight&seed=3"
                       "&wind=0.5&noise=2")
    assert status == 200
    records = [json.loads(line[len("data: "):]) for line in body.decode().splitlines()
               if line.startswith("data: ")]
    assert records[0]["type"] == "header"
    assert records[0]["scenario"] == "s2_crosswind_w0.5_n2"
    assert records[-1]["type"] == "summary"
    assert all(r["type"] == "tick" for r in records[1:-1])
    assert (runs / "s2_crosswind_w0.5_n2__straight__s3.jsonl").exists()


def test_live_rejects_bad_input(server):
    base, _ = server
    assert get(f"{base}/api/live?scenario=s2_crosswind&controller=pid&seed=1&wind=9")[0] == 400
    assert get(f"{base}/api/live?scenario=s2_crosswind&controller=pid&seed=1&noise=x")[0] == 400
    assert get(f"{base}/api/live?scenario=s5_noise_sweep&controller=pid&seed=1")[0] == 400
    assert get(f"{base}/api/live?scenario=s6_unseen_wind&controller=pid&seed=1")[0] == 400


def test_options_lists_controllers(server):
    base, _ = server
    status, body = get(f"{base}/api/options")
    options = json.loads(body)
    assert status == 200
    assert "shuffled" in options["controllers"] and "pid" in options["controllers"]
    assert "s5_noise_sweep" not in options["scenarios"]
