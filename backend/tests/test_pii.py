"""Measured recall and false positives of the regex PII detector on a small labelled SYNTHETIC set.

All names, emails and numbers below are invented for testing. The set deliberately
includes formats the detector is known to miss, so the measured false-negative rate
is honest (docs/adr/0005). Run `pytest backend/tests/test_pii.py -s` to print the table.
"""

from collections import Counter

from app.profile.pii import detect, mask

# (resume snippet, PII types it contains)
LABELLED = [
    ("Asha Kulkarni\nSkills: Python", {"name"}),
    ("ROHAN MEHTA | rohan.mehta@example.com", {"name", "email"}),
    ("Priya S. Nair\n+91 98765 43210", {"name", "phone"}),
    ("Name: Kunal Desai\nB.E Computer", {"name"}),
    ("Full Name - Meera Iyer", {"name"}),
    ("asha kulkarni\npython developer", {"name"}),  # lowercase: expected miss
    ("Skills: Java\nAarav Shah is a student", {"name"}),  # not on the first line: expected miss
    ("Aarav\nData Analyst", {"name"}),  # one-word name: expected miss
    ("Contact: neha.joshi+jobs@example.co.in", {"email"}),
    ("EMAIL: VIKRAM.R@EXAMPLE.ORG", {"email"}),
    ("mail: asha [at] example [dot] com", {"email"}),  # obfuscated: expected miss
    ("Mobile: 9876543210", {"phone"}),
    ("Phone +91-98765-43210", {"phone"}),
    ("Tel (022) 2345 6789", {"phone"}),
    ("Call +1 (415) 555-0123", {"phone"}),
    ("Reach me at 98765 43210.", {"phone"}),  # sentence-final: missed before the Phase 5 fix
    ("Ph: 555-0123", {"phone"}),  # 7 digits: expected miss
    ("Address: 12 MG Road, Pune 411001", {"address"}),
    ("Flat 4, Shanti Nagar, Mumbai 400076", {"address"}),
    ("Permanent Address - 22 Lake View, Kochi", {"address"}),
    ("Lives at 5 Park Street, Kolkata", {"address"}),  # no label, no PIN: expected miss
    ("linkedin.com/in/asha-kulkarni-123", {"url"}),
    ("Portfolio: https://github.com/rohanm", {"url"}),
    ("www.meera-iyer.dev", {"url"}),
    ("DOB: 12/04/2002", {"dob"}),
    ("Date of Birth - 4 May 2001", {"dob"}),
    ("Born on 4 May 2001", {"dob"}),  # unlabelled: expected miss
]
CLEAN = [  # real-looking resume lines with no PII
    "Skills: Python, SQL, Power BI, Machine Learning",
    "B.E Computer Engineering 2019-2023, Mumbai University",
    "Education 2015-2016 2017-2018 2019-2020",
    "Data Analyst Intern at Example Analytics, 6 months",
    "Project: Crop Disease Detection using CNNs (accuracy 94%)",
    "Certifications: AWS Cloud Practitioner, Google Data Analytics",
    "Experience: 24 months of Java development",
    "Operating Systems Windows XP Linux",
    "SKILLSET Oracle DBA RMAN",
    "Education Details B.Tech Civil Engineering",
    "Achievements: SIH 2024 Grand Finale - 15/12/2024",
    "Technologies: Node.js, React.js, ASP.NET, C#",
]


def evaluate() -> tuple[dict[str, tuple[int, int]], int]:
    expected, found = Counter(), Counter()
    for text, types in LABELLED:
        detected = {f.type for f in detect(text)}
        for t in types:
            expected[t] += 1
            found[t] += t in detected
    false_positive_lines = sum(bool(detect(line)) for line in CLEAN)
    return {t: (found[t], expected[t]) for t in sorted(expected)}, false_positive_lines


def test_measured_recall_and_false_positives():
    recall, fp = evaluate()
    print("\nPII detector (pii-regex-v2) on the labelled synthetic set:")
    for t, (hit, total) in recall.items():
        print(f"  {t:8s} recall {hit}/{total} = {hit / total:.0%}   false-negative rate {1 - hit / total:.0%}")
    print(f"  false-positive lines: {fp}/{len(CLEAN)}")
    # Regression floors at the measured values. They are not claims of real-world accuracy.
    assert recall["email"][0] >= 2 and recall["phone"][0] >= 6 and recall["url"][0] == 3
    assert recall["name"][0] >= 5 and recall["address"][0] >= 3 and recall["dob"][0] >= 2
    assert fp == 0


def test_findings_carry_types_and_offsets_but_never_values():
    text = "Asha Kulkarni | asha.k@example.com"
    findings = detect(text)
    assert all(set(f.model_dump()) == {"field", "type", "start", "end"} for f in findings)
    assert mask(text, findings) == "[NAME] | [EMAIL]"
