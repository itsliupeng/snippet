
# https://github.com/wandb/wandb/issues/6184, must use version 0.15.6
from wandb.apis.importers import WandbParquetImporter

if __name__ == '__main__':
    importer = WandbParquetImporter(
        src_base_url="https://api.wandb.ai",
        src_api_key="3e47658fc3138ec8580d195368499dd05ac5c735",
        dst_base_url="https://api.wandb.ai",
        dst_api_key="e0166a1ef78308cd5178dbf2abd947fa0fb5b356",
    )

    # src_entity
    runs = importer.collect_runs("binary-object-02")
    x = None
    for x in runs:
        # only copy open-llama-repo-232
        if x.run.name == 'open-llama-repo-232':
            importer.import_run(x)



# import wandb
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="deepspeed")
#
# # Iterate over each row in the DataFrame
# for index, row in d.iterrows():
#     # Create a dictionary for the current row
#     data = {col: row[col] for col in d.columns}
#
#     # Log the data to WandB
#     wandb.log(data)
#
# # Close the WandB run
# wandb.finish()


# import wandb
# api = wandb.Api()
# run = api.run("binary-object-02/deepspeed/gm1xzd6h")