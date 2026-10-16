import pytest
from pydantic import ValidationError

from app.auth.schemas import PasswordResetIn, SignupCompleteIn, password_rules


@pytest.mark.parametrize("pw, ok", [
    ("Abc1234!", True), ("ABCDEFG!1", True),
    ("abc12345", False),   # no capital, no special character
    ("Abc12345", False),   # no special character
    ("abcdefg!1", False),  # no capital
    ("Abcdefg!", False),   # no number
    ("Ab1!", False),       # too short
    ("Abcdefg 1", False),  # a space is not a special character
])
def test_password_needs_capital_number_and_special_character(pw, ok):
    if ok:
        assert password_rules(pw) == pw
    else:
        with pytest.raises(ValueError):
            password_rules(pw)


def test_signup_and_reset_apply_the_same_rule():
    with pytest.raises(ValidationError):
        SignupCompleteIn(signup_token="t", password="secret123", full_name="A")
    with pytest.raises(ValidationError):
        PasswordResetIn(email="a@b.co", otp="123456", password="secret123")
    assert PasswordResetIn(email="A@B.co", otp="123456", password="Secret#12").email == "a@b.co"
