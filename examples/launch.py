from hamilton import driver, base

import train_qm9

config = {}
adapter = base.SimplePythonGraphAdapter()

dr = driver.Driver(config, train_qm9, adapter=adapter)
out = dr.execute(final_vars=['train_qm9'], inputs={})