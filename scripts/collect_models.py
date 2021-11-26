import os
from google_drive_downloader import GoogleDriveDownloader as gdd

gids = {
    'cifar-10-swin-t-baseline.pth':'1xVkC1IUXKy4xa5VghzzM7sFewPweUSCJ',
    'cifar-10-swin-t-ours.pth': '1JL65kelWYztOH-SBfVeLIdv-GncfXNfA',
    'cifar-100-swin-t-baseline.pth': '1HnYL1UHX3a-thvLF-2JH09lly-gRs8aU',
    'cifar-100-swin-t-ours.pth': '1wOls3cBaXZtWiioSsjtXC3AUiU0rxyLU',
    'clipart-swin-t-baseline.pth': '1NnB1wpBD66n8e68QqEKtMZFR0f_949E7',
    'clipart-swin-t-ours.pth': '10uSG_3SJR3SEm6O46IYj_RRKRghPa0U_',
    'flowers-swin-t-baseline.pth': '1vetb-JV9igCZVsHTWlK0Qs8okVnBYuof',
    'flowers-swin-t-ours.pth': '1TaI8nY9J8qDVvyjYiqJ5NU5tjCNtJHzS',
    'infograph-swin-t-baseline.pth': '1zKKSXgP6NNgvwIfRD82ZCSAxrWB9Afz9',
    'infograph-swin-t-ours.pth': '1XBCw4UEA4TIg3R60Hll8i_L5rFpo9-Ql',
    'painting-swin-t-baseline.pth': '1UxBHysuzlDqk-zVnj87NhndH4xEoV0hF',
    'painting-swin-t-ours.pth': '1MAtYTRIsiHnabAba2YC7EF73GhmYzdZf',
    'quickdraw-swin-t-baseline.pth': '1fB6V87elDmuTobuG2_1w1HakU0GQLMav',
    'quickdraw-swin-t-ours.pth': '1JuxaLOJS8NvTpW--tUxvt_Nfe2-iOi4_',
    'real-swin-t-baseline.pth': '1JEWAYdRQXWbW9SoP3LQ2P_jCtzJLnJfS',
    'real-swin-t-ours.pth': '1X81M2WcdhA9Tt9kAAkLZ8amjTGuxXyNJ',
    'sketch-swin-t-baseline.pth': '1bXnv_jq_XdhUiDFm3XLOFPr3Kd_gQqP_',
    'sketch-swin-t-ours.pth': '1lAX5woCu1Fy8UVhV53kpzbqf4ESe8yDF',
    'svhn-swin-t-baseline.pth': '1WVmT_v0CJnLYsxO8yndOxLHFE8B0WbIo',
    'svhn-swin-t-ours.pth': '1Az5FuHAol6_o2gufPW2Fm2f6mKag-kRs'
}


save_dir = "../pretrained_models"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for f in gids:
    gdd.download_file_from_google_drive(file_id=gids[f], dest_path='{}/{}'.format(save_dir, f), unzip=False)
