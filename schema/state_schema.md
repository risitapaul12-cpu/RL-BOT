State Schema:-



This document defines the structure of the RL state vector.



The state vector must remain fixed in size and order.



State Structure:



The state vector is constructed as:



15 assets



15 features per asset



4 portfolio-level features appended at the end



Total dimension:



(15 × 15) + 4 = 229



Asset Ordering:



Assets must always appear in the same fixed order across all runs.



Example:



Asset 1



Asset 2

...



Asset 15



Asset order must never change.



Portfolio-Level Features (Appended at End):



cash\_ratio



portfolio\_drawdown



portfolio\_volatility



gross\_exposure

