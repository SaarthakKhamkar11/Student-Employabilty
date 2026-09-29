# Data Quality Report
## Jobs (sampled from real job_descriptions.csv)
- Row count: 4990
- Duplicate job_id: 0
- Missing fields: {'job_id': 0, 'source_job_id': 0, 'experience_required': 0, 'qualification_required': 0, 'salary_range': 0, 'location': 0, 'country': 0, 'work_type': 0, 'job_title': 0, 'role': 0, 'job_portal': 0, 'job_description': 0, 'skills': 0, 'responsibilities': 0, 'company_name': 0, 'gender_preference': 0}
- gender_preference distribution: {'both': 1685, 'male': 1663, 'female': 1642}

## Students (real, unified)
- Row count: 286
- Duplicate student_id: 0
- Rows by source_dataset: {'kaggle_resume_dataset': 166, 'placement_cell': 120}
- Rows by branch: {'Computer Science and Engineering (Data Science)': 130, 'Information Technology': 90, 'Non-Engineering / Management': 50, 'Civil Engineering': 6, 'Mechanical Engineering': 5, 'Electrical Engineering': 5}
- Missing fields (blank, never fabricated): {'student_id': 0, 'degree': 65, 'branch': 0, 'graduation_year': 166, 'gender': 166, 'avg_cgpa_sem5': 166, 'achievements': 166, 'career_interests': 0, 'domain': 0, 'internship_raw': 0, 'technical_projects': 166, 'skills_raw': 33, 'certifications': 166, 'country': 166, 'source_dataset': 0, 'is_synthetic': 0, 'synthetic_fields': 286}
