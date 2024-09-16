examples = [
    {
        "question": "Who are the customers that left Charter due to unresolved issues",
        "sql": "SELECT COUNT(DISTINCT t.`Account_Number`) AS `Customers_Left_Charter`, t.`Account_Number`, a.`Customer_Name` FROM `techorder` t JOIN `account` a ON t.`Account_Number` = a.`Account_Number` WHERE t.`Resolved_Postponed` = 'N' AND a.`End_Date` <> '1/1/9999' GROUP BY t.`Account_Number`, a.`Customer_Name`;"
    },
    {
        "question": "Why did they leave Charter?",
        "sql": "SELECT t.`Account_Number`, a.`Customer_Name`, MAX(t.`Disconnect_Reason`) AS `Disconnect_Reason` FROM `techorder` t JOIN `account` a ON t.`Account_Number` = a.`Account_Number` WHERE t.`Resolved_Postponed` = 'N' AND a.`End_Date` <> '1/1/9999' GROUP BY t.`Account_Number`, a.`Customer_Name`;"
    }
]
