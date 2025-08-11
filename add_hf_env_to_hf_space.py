from huggingface_hub import HfApi
import os
repo_id = "miladalsh/explaining_authorship_attribution_models"
api = HfApi()

api.add_space_variable(repo_id=repo_id, key="OPENAI_API_KEY", value=os.environ["OPENAI_API_KEY"])