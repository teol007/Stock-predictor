import sys

import great_expectations as gx

context = gx.get_context()

datasource_name = "AAPL_price"
data_asset_name = "AAPL_price_data"

# Load the data asset
asset = context.get_datasource(datasource_name).get_asset(data_asset_name)

# Load checkpoint
checkpoint_name = "AAPL_price_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)

# Run the checkpoint
run_id = "AAPL_price_run"
checkpoint_result = checkpoint.run(
    run_id=run_id
)

# Build data docs
context.build_data_docs()

# Check if the checkpoint passed
if checkpoint_result["success"]:
    print("Validation passed!")
    sys.exit(0)
else:
    print("Validation failed!")
    sys.exit(1)
