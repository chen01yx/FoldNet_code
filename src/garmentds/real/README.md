# real image collection

1. run `python run/real_image_collect.py category=tshirt clothes_name=tshirt_ss_0`
2. run `python src/garmentds/real/data_script/move_data_together.py -f yyyy-mm-dd`
3. run `python src/garmentds/real/human_label_app.py --category tshirt --img_dir data/keypoints/yyyy-mm-dd`
4. run `python src/garmentds/real/human_mask_app.py --img_dir data/keypoints/yyyy-mm-dd`