# Auth module: email-OTP signup, email + password login (JWT)

## Flow

**Signup (new user)**
1. `POST /auth/signup/request-otp` `{ "email": "...", "role": "student" | "recruiter" }`
   → a 6-digit code is emailed (valid 10 min). Returns 409 if the email already has an account.
2. `POST /auth/signup/verify-otp` `{ "email": "...", "otp": "123456" }`
   → `{ "signup_token": "...", "role": "...", "expires_in": 900 }`
3. `POST /auth/signup/complete`
   ```json
   {
     "signup_token": "<from step 2>",
     "password": "mypass123",
     "full_name": "Riya Shah",
     "student_profile": { "degree": "B.Tech", "branch": "CSE (Data Science)",
                          "graduation_year": 2027, "gender": "female" }
   }
   ```
   Recruiters send `"recruiter_profile": { "company_name": "Acme", "designation": "HR Lead" }` instead.
   → account created, returns an `access_token` so the user is logged in straight away.

**Login (existing user, no OTP)**
- `POST /auth/login` `{ "email": "...", "password": "..." }` → `{ "access_token": "...", "role": "...", "expires_in": 86400 }`
- The frontend sends `Authorization: Bearer <access_token>` on every protected request.
- `GET /auth/me` returns the logged-in user and their profile.

**Built-in protections**
- Only a hash of the OTP is stored (in Redis), never the code itself. Each OTP works once and is burned after 5 wrong tries.
- Resending is allowed after 60 s, with a maximum of 5 codes per email per hour.
- The signup token works once, and it can't be used as a login token (or the other way round).
- Passwords are stored as bcrypt hashes. They need at least 8 characters with at least one letter and one number.
- Login locks for 15 min after 10 wrong passwords. The error message never reveals whether an email is registered.
- Emails are lowercased, so `Riya@Gmail.com` and `riya@gmail.com` are the same account.
- `gender` must be one of `male`, `female`, `other`, `prefer_not_to_say` (lowercase).

---

## Setup

### 1. Put the folder in your backend
Copy the whole `auth/` folder next to the file that contains `app = FastAPI(...)`.
If you don't know which file that is, search the backend for `FastAPI(`.
(In this repo: `backend/app/auth/`, imported as `app.auth` from `backend/app/main.py`.)

### 2. Add the libraries to the backend `requirements.txt`
```
PyJWT==2.14.0               # Auth: JWT access + signup tokens
bcrypt>=4.2                 # Auth: password hashing
redis>=5.0                  # Auth: OTP storage, rate limits
email-validator>=2.2        # Auth: EmailStr validation
```
The rest of your file pins exact versions. After the rebuild in step 5, run
`docker compose exec backend pip freeze | grep -iE "bcrypt|redis|email.validator"` and pin those exact versions too.

### 3. Add to `.env`
```
JWT_SECRET=<python -c "import secrets; print(secrets.token_hex(32))">
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=<the Gmail address you created the App Password on>
SMTP_PASSWORD=<16-char App Password, no spaces>
SMTP_FROM_NAME=Employability Platform
OTP_EXPIRY_SECONDS=600
OTP_MAX_ATTEMPTS=5
```
Optional overrides (defaults shown): `ACCESS_TOKEN_EXPIRE_MINUTES=1440`, `SIGNUP_TOKEN_EXPIRE_MINUTES=15`,
`OTP_RESEND_COOLDOWN_SECONDS=60`, `OTP_MAX_SENDS_PER_HOUR=5`.

**Testing without email:** leave `SMTP_USER` empty while `ENVIRONMENT=development`. The OTP is then printed in
`docker compose logs -f backend` instead of being emailed.

Check that `docker-compose.yml` passes `.env` to the backend service (`env_file: .env` under the backend service).
If it lists variables one by one under `environment:` instead, add the new ones there.

### 4. Register the routes in your main file
```python
from auth import router as auth_router, init_auth_schema
# if your main file is inside a package (e.g. app/main.py) use:  from .auth import router as auth_router, init_auth_schema

app.include_router(auth_router)
```
To create the tables automatically when the server starts, call `init_auth_schema()` once at startup.
If you already have a `lifespan` function, put the call there. Otherwise add:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    init_auth_schema()          # creates app_users, student_accounts, recruiter_accounts if missing
    yield

app = FastAPI(lifespan=lifespan)   # add lifespan=... to your existing FastAPI(...) call
```

### 5. Rebuild and start
The requirements changed, so rebuild the image:
```
docker compose up -d --build backend
docker compose logs -f backend
```

### 6. Test in the browser
Open `http://localhost:<backend port>/docs` (Swagger UI) and run the endpoints in order:
`request-otp` → check your inbox → `verify-otp` → `complete` → `login`.
Copy the `access_token`, click **Authorize** (top right), paste it, then run `GET /auth/me`.

---

## Protecting your other routes
```python
from fastapi import Depends
from auth import get_current_user, require_role

@app.get("/students/me/matches")
def my_matches(user = Depends(require_role("student"))):
    # user = {"user_id": ..., "email": ..., "role": "student", "full_name": ...}
    ...

@app.post("/jobs")
def create_job(user = Depends(require_role("recruiter"))):
    ...
```

## Tables created
- `app_users`: email, bcrypt password hash, role, full name, active flag, timestamps
- `student_accounts`: degree, branch, graduation_year, gender, and `student_id`
- `recruiter_accounts`: company_name, designation

`student_accounts.student_id` stays empty at signup. Fill it in with the project-assigned student ID when that
student's record is created in your matching pipeline (for example, after their resume is parsed). That links
the login account to the student's matches.
