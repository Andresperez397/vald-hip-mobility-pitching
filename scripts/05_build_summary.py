"""Build the two-page summary (HTML, then PDF with headless Chrome) from the result tables.

python scripts/05_build_summary.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
NAME = "Hip Mobility and Pitching - Summary"


def main() -> int:
    r = json.loads((ROOT / "reports/tables/results.json").read_text())
    a = json.loads((ROOT / "reports/tables/data_audit.json").read_text())
    u = json.loads((ROOT / "reports/tables/uncleaned_comparison.json").read_text())
    sp, tq, h = r["release_speed"]["models"], r["elbow_torque"]["models"], r["hip_mocap"]
    html = (ROOT / "reports/summary_template.html").read_text()
    vals = {
        "RAW_SPEED": f"{u['ball_release_speed_mph']['body_size_and_handedness']['cv_r2']:.2f}",
        "RAW_TORQUE": f"{u['max_elbow_varus_torque_nm']['body_size_and_handedness']['cv_r2']:.2f}",
        "CLEAN_SPEED": f"{sp['body_size_and_handedness']['cv_r2']:.2f}",
        "CLEAN_TORQUE": f"{tq['body_size_and_handedness']['cv_r2']:.2f}",
        "SPEED_BASE": f"{sp['body_size_and_handedness']['cv_r2']:.3f}",
        "SPEED_HIP": f"{sp['plus_hip_tests']['cv_r2']:.3f}",
        "SPEED_ALL": f"{sp['plus_hip_and_shoulder_tests']['cv_r2']:.3f}",
        "TORQUE_BASE": f"{tq['body_size_and_handedness']['cv_r2']:.3f}",
        "TORQUE_HIP": f"{tq['plus_hip_tests']['cv_r2']:.3f}",
        "TORQUE_ALL": f"{tq['plus_hip_and_shoulder_tests']['cv_r2']:.3f}",
        "N_ROWS": str(a["rows"]),
        "N_CORRUPT": str(a["corrupt_rows"]),
        "N_PLACEHOLDER": str(a["placeholder_mass_rows"]),
        "N_SPEED": str(a["rows_with_usable_speed"]),
        "N_TORQUE": str(a["rows_with_usable_torque"]),
        "N_TARGETS": str(h["targets"]),
        "BEST_GAIN": f"{h['best_gain']:.2f}",
        "NULL_P95": f"{h['null_best_gain_p95']:.2f}",
        "P_BEST": f"{h['p_value_best_target']:.3f}",
        "SPEED_GAIN_CI": "{:.2f} to {:+.2f}".format(*sp["plus_hip_tests"]["gain_ci_first_repeat"]),
        "TORQUE_GAIN_CI": "{:.2f} to {:+.2f}".format(*tq["plus_hip_tests"]["gain_ci_first_repeat"]),
    }
    for k, v in vals.items():
        html = html.replace("{{" + k + "}}", v)
    out = ROOT / "reports" / "summary.html"
    out.write_text(html)
    pdf = ROOT / "reports" / f"{NAME}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{out}"],
                   check=True, capture_output=True)  # fmt: skip
    print("wrote", pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
