# ການຮັບຮູ້ການຂຽນທາງອາກາດພາສາລາວໂດຍໃຊ້ວິທີການຮຽນຮູ້ແບບເລິກເຊິ່ງ

## ກ່ຽວກັບໂຄງການ

ໂຄງການນີ້ແມ່ນບົດຈົບຊັ້ນປະລິນຍາຕີ ສາຂາວິທະຍາສາດຄອມພິວເຕີ ທີ່ພັດທະນາລະບົບການຮັບຮູ້ຕົວອັກສອນລາວໂດຍໃຊ້ທ່າທາງການຂຽນດ້ວຍມືເທິງອາກາດ. ລະບົບນີ້ໃຊ້ Computer Vision ສຳລັບການຕິດຕາມມື ແລະ ການຮັບຮູ້ທ່າທາງ ແລະ Deep Learning (CNN) ສຳລັບການຮັບຮູ້ຕົວອັກສອນ.

## ລະບົບປະຕິບັດການທີ່ຕ້ອງໃຊ້

**ລະບົບປະຕິບັດການ**: Linux (64-bit) - ໂຄງການນີ້ຖືກອອກແບບມາສະເພາະສຳລັບສະພາບແວດລ້ອມ Linux ແລະ ຕ້ອງການ dependencies ທີ່ສະເພາະເຈາະຈົງກັບ Linux.

### ຄຸນສົມບັດຫຼັກ

-   ການຕິດຕາມທ່າທາງມືແບບ Real-time ສຳລັບການຂຽນເທິງອາກາດ
-   ການຮັບຮູ້ຕົວອັກສອນລາວໂດຍໃຊ້ Convolutional Neural Networks (CNN)
-   ຮອງຮັບທັງສະຫຼະ ແລະ ພະຍັນຊະນະພາສາລາວ
-   ໜ້າຕ່າງຕິດຕໍ່ຜູ້ໃຊ້ (GUI) ທີ່ໃຊ້ງານງ່າຍ ສ້າງດ້ວຍ Tkinter

### Demo

![Lao Air-Writing Demo](src/assets/application_demo.gif)

## ໂຄງສ້າງຂອງໂຄງການ

```md
/
├── datasets/              # ຊຸດຂໍ້ມູນສຳລັບເຝິກ ແລະ ທົດສອບ (ດາວໂຫຼດແຍກຕ່າງຫາກ)
├── model/                 # ແບບຈຳລອງທີ່ເຝິກແລ້ວ ແລະ ໄຟລ໌ທີ່ກ່ຽວຂ້ອງ
├── src/
│   ├── assets/            # ໄຟລ໌ (Fonts, Demo, MediaPipe models)
│   ├── augment_image/     # ເຄື່ອງມືສຳລັບການເພີ່ມຂໍ້ມູນຮູບພາບ (Data Augmentation)
│   ├── collect_data/      # ໜ້າຕ່າງສຳລັບການເກັບກຳຂໍ້ມູນ
│   ├── lao_air_writting/  # ໂມດູນຫຼັກຂອງແອັບພລິເຄຊັນ
│   └── utils/             # ໂມດູນຊ່ວຍເຫຼືອ ແລະ ຟັງຊັນຕ່າງໆ
```

## ການຕິດຕັ້ງ

### ສິ່ງທີ່ຕ້ອງມີກ່ອນ

-   uv

### ການຕິດຕັ້ງ

1.  ຕິດຕັ້ງ uv ຖ້າທ່ານຍັງບໍ່ທັນມີ:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  Clone repository:

    ```bash
    git clone https://github.com/sila1404/lao_air_writting.git
    cd lao_air_writting
    ```

3.  ຕິດຕັ້ງ dependencies ໂດຍໃຊ້ uv:
    ```bash
    uv sync --extra local
    ```
    `--extra local` ຈະເພີ່ມ mediapipe/seaborn/scikit-learn/albumentations ທີ່ຈຳເປັນສຳລັບການເກັບກຳຂໍ້ມູນ, ເພີ່ມຂໍ້ມູນ, ເຝິກ, ແລະ ປະເມີນແບບຈຳລອງ. ຖ້າຕ້ອງການແຕ່ API server, ໃຫ້ໃຊ້ `--extra api` ແທນ (ແລະ `--extra postgres` ຖ້າໃຊ້ Postgres ສຳລັບເກັບ feedback) — ຈະຂ້າມ dependencies ທີ່ໃຊ້ສະເພາະ desktop/ການເຝິກ.
  
4. ດາວໂຫຼດຊຸດຂໍ້ມູນ (ເບິ່ງໃນພາກ [ຊຸດຂໍ້ມູນ](#ຊຸດຂໍ້ມູນ))

Dependencies ທີ່ຈຳເປັນທັງໝົດຖືກຈັດການໃນໄຟລ໌ `pyproject.toml`:

```toml
dependencies = [
"certifi",
"tensorflow>=2.19.0,<3",
"python-dotenv>=1.1.0,<2",
"torch>=2.7.1,<3",
"transformers>=4.52.4,<5",
"opencv-python-headless>=4.11.0,<5",
"numpy<2",
"pillow>=11.1.0,<12",
]
```
  
## ຊຸດຂໍ້ມູນ

ຊຸດຂໍ້ມູນສຳລັບການເຝິກ ແລະ ທົດສອບແບບຈຳລອງສາມາດດາວໂຫຼດໄດ້ຈາກແຫຼ່ງຕໍ່ໄປນີ້:

- **Hugging Face**: https://huggingface.co/datasets/silamany/lao-character-images
- **Kaggle**: https://www.kaggle.com/datasets/silamany/lao-characters

ຫຼັງຈາກດາວໂຫຼດແລ້ວ, ໃຫ້ແຕກໄຟລ໌ຊຸດຂໍ້ມູນລົງໃນໂຟເດີ datasets/ ທີ່ຢູ່ໃນ root directory ຂອງໂຄງການ.

## ການນຳໃຊ້

ໂຄງການປະກອບມີ 7 ຄຳສັ່ງຫຼັກສຳລັບຂັ້ນຕອນຕ່າງໆ:

### ການເກັບກຳຂໍ້ມູນ ແລະ ການເພີ່ມຂໍ້ມູນ (Data Collection and Augmentation)

-   ເກັບກຳຂໍ້ມູນ (Collect Data)

    ```bash
    uv run python src/collect_data/main.py
    ```

    -   ເປີດໜ້າຕ່າງສຳລັບການເກັບກຳຂໍ້ມູນ
    -   ໃຊ້ທ່າທາງມືເພື່ອຂຽນຕົວອັກສອນລາວ
    -   ຕົວອັກສອນຈະຖືກບັນທຶກໄວ້ໃນໂຟເດີສະຫຼະ/ພະຍັນຊະນະຕາມລຳດັບ

-   ເພີ່ມຂໍ້ມູນ (Augment Data)
    ```bash
    uv run python src/augment_image/main.py
    ```
    -   ທຳການເພີ່ມຂໍ້ມູນ (data augmentation) ໃສ່ຮູບພາບທີ່ເກັບກຳມາ
    -   ເພີ່ມຂະໜາດຂອງຊຸດຂໍ້ມູນຜ່ານການປ່ຽນແປງຮູບແບບຕ່າງໆ
    -   ຊ່ວຍປັບປຸງຄວາມທົນທານຂອງແບບຈຳລອງ

### ການເຝິກແບບຈຳລອງ (Model Training)

-   ແບ່ງຊຸດຂໍ້ມູນ (Split Dataset)

    ```bash
    uv run python src/augment_image/split_data.py
    ```

    -   ແບ່ງຂໍ້ມູນທີ່ເກັບກຳມາອອກເປັນຊຸດຂໍ້ມູນສຳລັບເຝິກ (training) ແລະ ທົດສອບ (testing)
    -   ກຽມຂໍ້ມູນສຳລັບການເຝິກແບບຈຳລອງ

-   ເຝິກແບບຈຳລອງ (Train Model)

    ```bash
    uv run python src/lao_air_writting/train_model.py
    ```

    -   ເລີ່ມຕົ້ນຂະບວນການເຝິກແບບຈຳລອງ CNN
    -   ໃຊ້ຊຸດຂໍ້ມູນສຳລັບເຝິກທີ່ກຽມໄວ້
    -   ບັນທຶກແບບຈຳລອງທີ່ເຝິກສຳເລັດແລ້ວ

-   ປະເມີນແບບຈຳລອງ (Evaluate Model)

    ```bash
    uv run python src/lao_air_writting/evaluate_model.py
    ```

    -   ປະເມີນປະສິດທິພາບຂອງແບບຈຳລອງທີ່ເຝິກແລ້ວ
    -   ສ້າງຄ່າວັດແທກປະສິດທິພາບ ແລະ ລາຍງານຜົນ

-   ທົດສອບແບບຈຳລອງ (Test Model)
    ```bash
    uv run python src/lao_air_writting/test_app.py
    ```
    -   ເປີດໜ້າຕ່າງຫຼັກຂອງແອັບພລິເຄຊັນ
    -   ສາມາດຂຽນ ແລະ ຮັບຮູ້ຕົວອັກສອນແບບ Real-time

### API Server

-   ເລີ່ມ API Server
    ```bash
    uv run --extra api uvicorn lao_air_writting.api:app
    ```
    -   ເປີດ API server ສໍາລັບການຮັບຮູ້ຕົວອັກສອນລາວ
    -   ສະຫນອງ endpoints ສໍາລັບການຮັບຮູ້ຂໍ້ຄວາມ
    -   Server ເຮັດວຽກຢູ່ localhost (port ເລີ່ມຕົ້ນ: 8000)

## ການແກ້ໄຂບັນຫາ

ຖ້າທ່ານພົບບັນຫາຕໍ່ໄປນີ້:

_ModuleNotFoundError: No module named 'certifi'_

ທ່ານສາມາດແກ້ໄຂໄດ້ໂດຍການລຶບ virtual environment:

```bash
rm -rf .venv
```

ຈາກນັ້ນ, ຕິດຕັ້ງ dependencies ຄືນໃໝ່:

```bash
uv sync --extra local
```

## ຫຼັກການເຮັດວຽກ

-   **ການຕິດຕາມມື (Hand Tracking)**: ໃຊ້ MediaPipe ສຳລັບການກວດຈັບຈຸດສຳຄັນເທິງມືແບບ Real-time
-   **ການແຕ້ມຕົວອັກສອນ (Character Drawing)**: ຕິດຕາມການເຄື່ອນໄຫວຂອງນິ້ວຊີ້ເພື່ອສ້າງຮູບແຕ້ມຕົວອັກສອນ
-   **ການຮັບຮູ້ (Recognition)**: ປະມວນຜົນຮູບແຕ້ມຜ່ານແບບຈຳລອງ CNN ທີ່ເຝິກແລ້ວ

## ສະຖາປັດຕະຍະກຳຂອງແບບຈຳລອງ (Model Architecture)

ແບບຈຳລອງການຮັບຮູ້ຕົວອັກສອນໃຊ້ສະຖາປັດຕະຍະກຳແບບ Convolutional Neural Network (CNN):

-   ຊັ້ນ Input ສຳລັບປະມວນຜົນຮູບພາບຕົວອັກສອນ
-   ຫຼາຍຊັ້ນ Convolutional ແລະ Pooling
-   ຊັ້ນ Dense ສຳລັບການຈັດປະເພດ
-   ຊັ້ນ Output ສຳລັບການຮັບຮູ້ຕົວອັກສອນລາວ

## ຜູ້ຂຽນ

ສີລາມະນີ ໂຮມພະສະຖານ & ພົງສະຫວັນ ແສງອົກປະດິດ  
ພາກວິຊາວິທະຍາສາດຄອມພິວເຕີ  
ຄະນະວິທະຍາສາດທຳມະຊາດ
ມະຫາວິທະຍາໄລແຫ່ງຊາດລາວ

## ຂໍຂອບໃຈ

-   ອາຈານທີ່ປຶກສາ

    -   ດຣ. ສົມສັກ ອິນທະສອນ
    -   ອຈ.ປທ. ສົມມະນີ ລູຊະວົງ

-   ອາສາສະໝັກເກັບກຳຂໍ້ມູນ  
    ພວກເຮົາຂໍສະແດງຄວາມຂອບໃຈຢ່າງຈິງໃຈມາຍັງອາສາສະໝັກທຸກທ່ານທີ່ໄດ້ສະຫຼະເວລາ ແລະ ເຫື່ອແຮງໃນການສະໜອງຕົວຢ່າງລາຍມືສຳລັບຊຸດຂໍ້ມູນຂອງພວກເຮົາ:

    -   ນັກສຶກສາຈາກ ສາຂາພັດທະນາໂປຣແກຣມ, ສະຖາບັນເຕັກໂນໂລຊີ ສຸດສະກະ
    -   ສະມາຊິກຈາກ ພາກວິຊາວິທະຍາສາດຄອມພິວເຕີ

    ການປະກອບສ່ວນຂອງພວກທ່ານແມ່ນສິ່ງສຳຄັນທີ່ສຸດໃນການສ້າງຊຸດຂໍ້ມູນທີ່ຫຼາກຫຼາຍ ແລະ ຄົບຖ້ວນສຳລັບການເຝິກແບບຈຳລອງຂອງພວກເຮົາ.
