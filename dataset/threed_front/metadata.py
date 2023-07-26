#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

THREED_FRONT_BEDROOM_FURNITURE_MAP = {
    "desk": "desk",
    "nightstand": "nightstand",
    "king-size bed": "double_bed",
    "single bed": "single_bed",
    "kids bed": "kids_bed",
    "ceiling lamp": "ceiling_lamp",
    "pendant lamp": "pendant_lamp",
    "bookcase/jewelry armoire": "bookshelf",
    "tv stand": "tv_stand",
    "wardrobe": "wardrobe",
    "lounge chair/cafe chair/office chair": "chair",
    "dining chair": "chair",
    "classic chinese chair": "chair",
    "armchair": "armchair",
    "dressing table": "dressing_table",
    "dressing chair": "dressing_chair",
    "corner/side table": "table",
    "dining table": "table",
    "round end table": "table",
    "drawer chest/corner cabinet": "cabinet",
    "sideboard/side cabinet/console table": "cabinet",
    "children cabinet": "children_cabinet",
    "shelf": "shelf",
    "footstool/sofastool/bed end stool/stool": "stool",
    "coffee table": "coffee_table",
    "loveseat sofa": "sofa",
    "three-seat/multi-seat sofa": "sofa",
    "l-shaped sofa": "sofa",
    "lazy sofa": "sofa",
    "chaise longue sofa": "sofa",
}

THREED_FRONT_BEDROOM_FURNITURE = [
    'armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'children_cabinet', 'coffee_table', 'desk', 'door',
    'double_bed', 'dressing_chair', 'dressing_table', 'kids_bed', 'nightstand', 'pendant_lamp', 'shelf', 'single_bed',
    'sofa', 'stool', 'table', 'tv_stand', 'wall', 'wardrobe', 'window', 'empty'
]
THREED_FRONT_BEDROOM_FURNITURE_CNTS = {
    "wall": 11072,
    "nightstand": 2931,
    "door": 2644,
    "double_bed": 1910,
    "window": 1798,
    "wardrobe": 1749,
    "pendant_lamp": 1379,
    "ceiling_lamp": 731,
    "tv_stand": 333,
    "chair": 271,
    "dressing_table": 239,
    "single_bed": 225,
    "table": 218,
    "cabinet": 214,
    "desk": 201,
    "stool": 127,
    "kids_bed": 93,
    "bookshelf": 90,
    "children_cabinet": 88,
    "shelf": 86,
    "dressing_chair": 64,
    "armchair": 40,
    "sofa": 21,
    "coffee_table": 17
}

THREED_FRONT_BEDROOM_MIN_FURNITURE_NUM = 3
THREED_FRONT_BEDROOM_MAX_FURNITURE_NUM = 13
THREED_FRONT_BEDROOM_MIN_WALL_NUM = 4
THREED_FRONT_BEDROOM_MAX_WALL_NUM = 10

THREED_FRONT_LIBRARY_FURNITURE_MAP = {
    "bookcase/jewelry armoire": "bookshelf",
    "desk": "desk",
    "pendant lamp": "pendant_lamp",
    "ceiling lamp": "ceiling_lamp",
    "lounge chair/cafe chair/office chair": "lounge_chair",
    "dining chair": "dining_chair",
    "dining table": "dining_table",
    "corner/side table": "corner_side_table",
    "classic chinese chair": "chinese_chair",
    "armchair": "armchair",
    "shelf": "shelf",
    "sideboard/side cabinet/console table": "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool": "stool",
    "round end table": "round_end_table",
    "loveseat sofa": "loveseat_sofa",
    "drawer chest/corner cabinet": "cabinet",
    "wardrobe": "wardrobe",
    "three-seat/multi-seat sofa": "multi_seat_sofa",
    "wine cabinet": "wine_cabinet",
    "coffee table": "coffee_table",
    "lazy sofa": "lazy_sofa",
    "children cabinet": "cabinet",
    "chaise longue sofa": "chaise_longue_sofa",
    "l-shaped sofa": "l_shaped_sofa",
    "dressing table": "dressing_table",
    "dressing chair": "dressing_chair",
}

THREED_FRONT_LIBRARY_FURNITURE = [
    'armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'children_cabinet', 'coffee_table', 'desk', 'door',
    'double_bed', 'dressing_chair', 'dressing_table', 'kids_bed', 'nightstand', 'pendant_lamp', 'shelf', 'single_bed',
    'sofa', 'stool', 'table', 'tv_stand', 'wall', 'wardrobe', 'window', 'empty'
]
THREED_FRONT_LIBRARY_FURNITURE_CNTS = {
    "wall": 11072,
    "nightstand": 2931,
    "door": 2644,
    "double_bed": 1910,
    "window": 1798,
    "wardrobe": 1749,
    "pendant_lamp": 1379,
    "ceiling_lamp": 731,
    "tv_stand": 333,
    "chair": 271,
    "dressing_table": 239,
    "single_bed": 225,
    "table": 218,
    "cabinet": 214,
    "desk": 201,
    "stool": 127,
    "kids_bed": 93,
    "bookshelf": 90,
    "children_cabinet": 88,
    "shelf": 86,
    "dressing_chair": 64,
    "armchair": 40,
    "sofa": 21,
    "coffee_table": 17
}

THREED_FRONT_LIBRARY_MIN_FURNITURE_NUM = 3
THREED_FRONT_LIBRARY_MAX_FURNITURE_NUM = 10
THREED_FRONT_LIBRARY_MIN_WALL_NUM = 4
THREED_FRONT_LIBRARY_MAX_WALL_NUM = 10

THREED_FRONT_LIVINGROOM_FURNITURE_MAP = {
    "bookcase/jewelry armoire": "bookshelf",
    "desk": "desk",
    "pendant lamp": "pendant_lamp",
    "ceiling lamp": "ceiling_lamp",
    "lounge chair/cafe chair/office chair": "lounge_chair",
    "dining chair": "dining_chair",
    "dining table": "dining_table",
    "corner/side table": "corner_side_table",
    "classic chinese chair": "chinese_chair",
    "armchair": "armchair",
    "shelf": "shelf",
    "sideboard/side cabinet/console table": "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool": "stool",
    "round end table": "round_end_table",
    "loveseat sofa": "loveseat_sofa",
    "drawer chest/corner cabinet": "cabinet",
    "wardrobe": "wardrobe",
    "three-seat/multi-seat sofa": "multi_seat_sofa",
    "wine cabinet": "wine_cabinet",
    "coffee table": "coffee_table",
    "lazy sofa": "lazy_sofa",
    "children cabinet": "cabinet",
    "chaise longue sofa": "chaise_longue_sofa",
    "l-shaped sofa": "l_shaped_sofa",
    "tv stand": "tv_stand"
}
THREED_FRONT_LIVINGROOM_FURNITURE = [
    'armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'children_cabinet', 'coffee_table', 'desk', 'door',
    'double_bed', 'dressing_chair', 'dressing_table', 'kids_bed', 'nightstand', 'pendant_lamp', 'shelf', 'single_bed',
    'sofa', 'stool', 'table', 'tv_stand', 'wall', 'wardrobe', 'window', 'empty'
]
THREED_FRONT_LIVINGROOM_FURNITURE_CNTS = {
    "wall": 11072,
    "nightstand": 2931,
    "door": 2644,
    "double_bed": 1910,
    "window": 1798,
    "wardrobe": 1749,
    "pendant_lamp": 1379,
    "ceiling_lamp": 731,
    "tv_stand": 333,
    "chair": 271,
    "dressing_table": 239,
    "single_bed": 225,
    "table": 218,
    "cabinet": 214,
    "desk": 201,
    "stool": 127,
    "kids_bed": 93,
    "bookshelf": 90,
    "children_cabinet": 88,
    "shelf": 86,
    "dressing_chair": 64,
    "armchair": 40,
    "sofa": 21,
    "coffee_table": 17
}

THREED_FRONT_LIVINGROOM_MIN_FURNITURE_NUM = 3
THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM = 21
THREED_FRONT_LIVINGROOM_MIN_WALL_NUM = 4
THREED_FRONT_LIVINGROOM_MAX_WALL_NUM = 20
