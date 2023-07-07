import os

from huggingface_hub import hf_hub_download

"""
Download mpt-30B-chat-GGML model from huggingface hub
"""


def download_mpt_model(destination_folder: str, repo_id: str, model_filename: str):
    local_path = os.path.abspath(destination_folder)
    return hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        local_dir=local_path,
        local_dir_use_symlinks=True,
    )


if __name__ == "__main__":
    """full url: https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q4_1.bin
    Name	Quant method	Bits	Size	Max RAM required	Use case
mpt-30b-chat.ggmlv0.q4_0.bin	q4_0	4	16.85 GB	19.35 GB	4-bit.
mpt-30b-chat.ggmlv0.q4_1.bin	q4_1	4	18.73 GB	21.23 GB	4-bit. Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.
mpt-30b-chat.ggmlv0.q5_0.bin	q5_0	5	20.60 GB	23.10 GB	5-bit. Higher accuracy, higher resource usage and slower inference.
mpt-30b-chat.ggmlv0.q5_1.bin	q5_1	5	22.47 GB	24.97 GB	5-bit. Even higher accuracy, resource usage and slower inference.
mpt-30b-chat.ggmlv0.q8_0.bin	q8_0	8	31.83 GB	34.33 GB	8-bit. Almost indistinguishable from float16. High resource use and slow. Not recommended for most users.
    
    """

    repo_id = "TheBloke/mpt-30B-chat-GGML"
    model_filename = "mpt-30b-chat.ggmlv0.q4_1.bin"
    destination_folder = "models"

    download_mpt_model(destination_folder, repo_id, model_filename)
    print("model downloaded")
