"""Run once to create the auth tables:  python -m app.auth.init_db"""
from .db import init_auth_schema

if __name__ == "__main__":
    init_auth_schema()
    print("Auth tables ready: app_users, student_accounts, recruiter_accounts")
