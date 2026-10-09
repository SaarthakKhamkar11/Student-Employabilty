"""The grading rubric shown on the labelling page and recorded with every judgment."""

RUBRIC_VERSION = "rubric-v1"
LABEL_SOURCE = "human_reviewed"

QUESTION = "Should this job appear in this student's job recommendations?"
GRADES = {
    3: ("Strong match", "The job's core skills and field match this student's skills, projects or "
                        "experience. They could credibly apply now."),
    2: ("Moderate match", "Same field and some core skills match, but there are clear gaps."),
    1: ("Weak match", "Related area only (e.g. a different specialty in the same industry); little skill overlap."),
    0: ("Irrelevant", "A different field: the student's background does not fit this job."),
}
IGNORE = [
    "salary, location, company and the employer's gender preference (not shown)",
    "years of experience and the degree a posting asks for (not shown: in this dataset they are "
    "assigned at random per posting, so they say nothing about the job)",
    "how good the student is overall (CGPA is not shown): judge the fit, not the student",
]
