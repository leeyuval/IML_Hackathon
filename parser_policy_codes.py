import pandas as pd
import re
"""
Hackathon 2023
- 
- 
- 
-

"""


def policy_parser(df: pd.DataFrame) -> pd.DataFrame:
    # Extract the cost values from the cancellation strings
    # days_pattern = r"(\d*)D(\d*)[NP]"
    # night_pattern = r"(\d*)N(\d*)[NP]"
    # no_show_pattern = r"_?(\d*)P$"
    #
    # # Extract the night charge using the night pattern
    # df['night_charge'] = df['cancellation_policy_code'].str.extract(night_pattern, expand=False)[1]
    # df['night_charge'].fillna(0, inplace=True)
    #
    # # Extract the days using the days pattern
    # df['days'] = df['cancellation_policy_code'].str.extract(days_pattern, expand=False)[0]
    # df['days'].fillna(0, inplace=True)
    #
    # # Extract the cost for cancellation using the days pattern
    # df['cost_cancellation'] = df['cancellation_policy_code'].str.extract(days_pattern, expand=False)[1]
    # df['cost_cancellation'] = df['cost_cancellation'].where(df['cancellation_policy_code'].str.contains('P'), 0)
    # df['cost_cancellation'].fillna(0, inplace=True)
    #
    # # Extract the cost for no-show using the no-show pattern
    # df['cost_no_show'] = df['cancellation_policy_code'].str.extract(no_show_pattern, expand=False)
    # df['cost_no_show'].fillna(0, inplace=True)

    policy_pattern = r"(\d+[DN])"
    no_show_pattern = r"_([NP]\d*)$"

    # Extract the policies using the policy pattern
    policies = df['cancellation_policy_code'].str.findall(policy_pattern)

    # Create a list to store the policy objects
    policy_objects = []

    # Process each policy in the list
    for policy_list in policies:
        policy_object = {}
        for policy in policy_list:
            days = re.findall(r"\d+", policy)[0]
            if 'D' in policy:
                policy_object['days_before'] = int(days)
            elif 'N' in policy:
                policy_object['night_charge'] = int(days)
        policy_objects.append(policy_object)

    # Extract the no-show policy if present
    no_show_policy = df['cancellation_policy_code'].str.extract(no_show_pattern, expand=False)
    if not no_show_policy.isna().values[0]:
        if no_show_policy.values[0][0] == 'P':
            policy_objects[-1]['no_show_policy'] = 'percentage_charge'
            policy_objects[-1]['percentage_charge'] = int(no_show_policy.values[0][1:])
        elif no_show_policy.values[0][0] == 'N':
            policy_objects[-1]['no_show_policy'] = 'night_charge'
            policy_objects[-1]['night_charge'] = int(no_show_policy.values[0][1:])
    return df


if __name__ == '__main__':
    pass
