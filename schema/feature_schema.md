Feature Schema:-



This document defines the required structure for asset-level feature CSV files.



All feature CSV files must strictly follow this format.



General Rules:



Use only past data (no future leakage).



Rolling normalization window: 60 days.



No global mean/std allowed.



Clip extreme values before saving.



Column order must never change.



Required Columns (In Exact Order):



ret\_1d



ret\_5d



ret\_10d



price\_vs\_ema20



ema20\_vs\_ema50



rsi



macd\_hist



atr\_norm



vol\_20d



volume\_ratio



obv\_slope



Note:

position, position\_size, unrealized\_pnl, and time\_in\_trade are generated inside the environment and must NOT be included in CSV files.

