import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "overtime": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "job_satisfaction": [1, 2, 3, 4, 1, 4],
            "monthly_income": [3000.0, 4000.0, 5000.0, 6000.0, 3500.0, 7000.0],
            "attrition": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "travel_frequency": ["Frequent", "Rarely", "Rarely", "Frequent", "Frequent", "Rarely"],
            "years_at_company": [1, 2, 5, 8, 1, 10],
            "age": [25, 30, 35, 40, 28, 45],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent(sample_df):
    # 3 leavers out of 6 employees = 50%
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = result.set_index("department")["attrition_rate"]
    assert rates["Sales"] == 100.0   # 2 out of 2 left
    assert rates["IT"] == 50.0       # 1 out of 2 left
    assert rates["HR"] == 0.0        # 0 out of 2 left


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = result.set_index("overtime")["attrition_rate"]
    assert rates["Yes"] == 100.0   # all 3 overtime employees left
    assert rates["No"] == 0.0      # no non-overtime employees left


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    income = result.set_index("attrition")["avg_monthly_income"]
    assert income["Yes"] == 3500.0   # (3000 + 4000 + 3500) / 3
    assert income["No"] == 6000.0    # (5000 + 6000 + 7000) / 3


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rates_per_group(sample_df):
    # Validates the denominator is per-group headcount, not total leavers
    result = satisfaction_summary(sample_df)
    rates = result.set_index("job_satisfaction")["attrition_rate"]
    assert rates[1] == 100.0   # 2 out of 2 employees with score 1 left
    assert rates[2] == 100.0   # 1 out of 1 employee with score 2 left
    assert rates[3] == 0.0     # 0 out of 1 employee with score 3 left
    assert rates[4] == 0.0     # 0 out of 2 employees with score 4 left


def test_satisfaction_summary_sorted_by_score(sample_df):
    result = satisfaction_summary(sample_df)
    scores = result["job_satisfaction"].tolist()
    assert scores == sorted(scores)
