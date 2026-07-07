import json

from stylereel.contract import (STYLES, Task, canonical_style, read_tasks,
                                write_results)


def test_read_tasks(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps([
        {"task_id": "v1", "video_url": "http://x/a.mp4",
         "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]},
        {"task_id": "v2", "video_url": "http://x/b.mp4"},
    ]))
    tasks = read_tasks(p)
    assert tasks[0].task_id == "v1"
    assert tasks[0].styles == list(STYLES)
    assert tasks[1].styles == list(STYLES)  # missing styles -> all four


def test_read_tasks_skips_malformed_entries(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps([
        {"task_id": "good", "video_url": "http://x/a.mp4"},
        {"task_id": "bad-no-url"},
        "not even a dict",
    ]))
    tasks = read_tasks(p)
    assert [t.task_id for t in tasks] == ["good"]


def test_read_tasks_accepts_wrapped_dict(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps({"tasks": [{"task_id": "v1", "video_url": "u"}]}))
    assert read_tasks(p)[0].task_id == "v1"


def test_unknown_style_key_preserved_verbatim(tmp_path):
    tasks = [Task("v1", "u", styles=["Formal", "poetic"])]
    write_results(tmp_path / "r.json", {"v1": {"Formal": "A caption."}}, tasks)
    data = json.loads((tmp_path / "r.json").read_text())
    caps = data[0]["captions"]
    assert set(caps) == {"Formal", "poetic"}  # exact requested keys
    assert caps["Formal"] == "A caption."
    assert caps["poetic"].strip()  # generic fallback filled


def test_canonical_style():
    assert canonical_style("Formal") == "formal"
    assert canonical_style("humorous-tech") == "humorous_tech"
    assert canonical_style("HUMOROUS NON TECH") == "humorous_non_tech"
    assert canonical_style("poetic") is None


def test_write_results_fills_holes(tmp_path):
    tasks = [Task("v1", "http://x/a.mp4"), Task("v2", "http://x/b.mp4")]
    results = {"v1": {"formal": "A caption.", "sarcastic": ""}}  # v2 missing entirely
    out_path = tmp_path / "results.json"
    write_results(out_path, results, tasks)
    data = json.loads(out_path.read_text())
    assert len(data) == 2
    by_id = {d["task_id"]: d["captions"] for d in data}
    assert by_id["v1"]["formal"] == "A caption."
    for style in STYLES:
        assert by_id["v1"][style].strip()
        assert by_id["v2"][style].strip()
    assert not (tmp_path / "results.json.tmp").exists()  # atomic write cleaned up


def test_write_results_garbage_values(tmp_path):
    tasks = [Task("v1", "u")]
    write_results(tmp_path / "r.json", {"v1": {"formal": None, "sarcastic": 42}}, tasks)
    data = json.loads((tmp_path / "r.json").read_text())
    assert all(isinstance(v, str) and v for v in data[0]["captions"].values())


def test_normalize_caption_strips_emdashes():
    from stylereel.contract import normalize_caption
    assert "—" not in normalize_caption("It does nothing—no blink, no twitch.")
    assert normalize_caption("It does nothing—no blink.") == "It does nothing, no blink."
    assert normalize_caption("sun, its flare — then it lands") == "sun, its flare, then it lands"
    # hyphenated words preserved
    assert "close-up" in normalize_caption("a close-up shot")
    # smart quotes normalized
    assert '"' not in normalize_caption("“quote”").replace('"', "")
