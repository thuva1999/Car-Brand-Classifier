import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from color_model import hsv_colour


def solid_hsv_image(hue: int, saturation: int, value: int) -> Image.Image:
    hsv = np.full((100, 100, 3), (hue, saturation, value), dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def striped_body(parts: list[tuple[tuple[int, int, int], int]]) -> Image.Image:
    """100x100 image whose body band (rows 30-70) is horizontal HSV stripes.

    ``parts`` is [(hsv, rows), ...] with rows summing to 40; everything
    outside the band repeats the first stripe."""

    hsv = np.full((100, 100, 3), parts[0][0], dtype=np.uint8)
    row = 30
    for colour, rows in parts:
        hsv[row:row + rows] = colour
        row += rows
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class HsvColourTests(unittest.TestCase):
    def test_votes_for_eight_exam_colours(self):
        examples = (
            ((0, 0, 240), "white"),
            ((0, 0, 25), "black"),
            ((0, 0, 120), "grey"),
            ((0, 200, 170), "red"),
            ((110, 200, 170), "blue"),
            ((60, 200, 170), "green"),
            ((18, 200, 120), "brown"),
            ((28, 200, 180), "yellow"),
        )

        for hsv, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(hsv_colour(solid_hsv_image(*hsv)), expected)

    def test_roi_excludes_roof_and_side_background(self):
        image = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
        image[30:70, 15:85] = (255, 255, 255)

        self.assertEqual(hsv_colour(Image.fromarray(image)), "white")

    def test_wheels_and_road_below_body_are_ignored(self):
        hsv = np.full((100, 100, 3), (110, 200, 170), dtype=np.uint8)
        hsv[70:] = (0, 0, 30)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.assertEqual(hsv_colour(Image.fromarray(rgb)), "blue")

    def test_small_coloured_patch_is_enough(self):
        hsv = np.full((100, 100, 3), (0, 0, 120), dtype=np.uint8)
        hsv[30:70, 15:85][:8] = (0, 200, 150)  # 20% of the body band
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.assertEqual(hsv_colour(Image.fromarray(rgb)), "red")

    def test_shadow_and_glare_do_not_outvote_coloured_body(self):
        hsv = np.full((100, 100, 3), (0, 200, 150), dtype=np.uint8)
        body = hsv[30:70, 15:85]
        body[:15] = (0, 0, 20)
        body[15:25] = (0, 0, 250)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.assertEqual(hsv_colour(Image.fromarray(rgb)), "red")

    def test_low_saturation_blue_reflections_do_not_turn_white_paint_blue(self):
        hsv = np.full((100, 100, 3), (0, 0, 240), dtype=np.uint8)
        body = hsv[30:70, 15:85]
        body[:, :20] = (110, 50, 150)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.assertEqual(hsv_colour(Image.fromarray(rgb)), "white")

    def test_glossy_black_paint_is_black_not_grey(self):
        # Real black cars in street photos have median V around 20-65.
        self.assertEqual(hsv_colour(solid_hsv_image(0, 0, 45)), "black")

    def test_dark_saturated_teal_paint_is_black_not_grey(self):
        # Live photo 16 Nov: dark green Leapmotor C16, median V ~88, S ~87-126,
        # hue ~98; too few pixels pass the blue vote (S >= 100).
        self.assertEqual(hsv_colour(solid_hsv_image(98, 90, 88)), "black")

    def test_reflective_black_with_bright_reflections_is_black(self):
        # Live cars #13/#17: dark core, reflections lift the median to ~110.
        image = striped_body([((0, 10, 30), 14), ((60, 30, 110), 12), ((100, 30, 180), 14)])
        self.assertEqual(hsv_colour(image), "black")

    def test_dark_body_with_blue_tinted_reflections_is_black_not_blue(self):
        # Live cars #27-#29, #42, #52: screen's blue cast on black paint.
        image = striped_body([((0, 0, 40), 16), ((110, 130, 120), 10), ((0, 0, 150), 14)])
        self.assertEqual(hsv_colour(image), "black")

    def test_vivid_blue_on_dark_body_stays_blue(self):
        # Live car #12: dark parts, but the blue itself is vivid (S ~196).
        image = striped_body([((0, 0, 40), 12), ((110, 200, 140), 16), ((0, 0, 180), 12)])
        self.assertEqual(hsv_colour(image), "blue")

    def test_lit_tail_lights_do_not_make_a_grey_car_red(self):
        # Live cars #21-#23: rear view, bright red tail lights (V ~200).
        image = striped_body([((0, 15, 150), 32), ((0, 200, 210), 8)])
        self.assertNotEqual(hsv_colour(image), "red")

    def test_dark_red_paint_is_still_red(self):
        # Live car #11: red paint is far darker than a lit tail light.
        image = striped_body([((0, 0, 40), 12), ((0, 150, 100), 16), ((0, 0, 160), 12)])
        self.assertEqual(hsv_colour(image), "red")

    def test_light_grey_body_with_dark_wheel_arch_stays_grey(self):
        # A dark arch in the band must not turn a mid-grey car black.
        image = striped_body([((0, 0, 30), 8), ((0, 10, 140), 32)])
        self.assertEqual(hsv_colour(image), "grey")

    def test_dark_grey_with_blue_cast_stays_grey(self):
        self.assertEqual(hsv_colour(solid_hsv_image(106, 45, 95)), "grey")

    def test_mid_neutral_paint_is_grey(self):
        self.assertEqual(hsv_colour(solid_hsv_image(0, 0, 90)), "grey")

    def test_camera_blue_cast_on_grey_paint_stays_grey(self):
        self.assertEqual(hsv_colour(solid_hsv_image(104, 50, 130)), "grey")

    def test_white_paint_in_shade_is_white(self):
        self.assertEqual(hsv_colour(solid_hsv_image(0, 15, 165)), "white")

    def test_silver_with_sky_tint_at_same_brightness_is_grey(self):
        self.assertEqual(hsv_colour(solid_hsv_image(104, 45, 165)), "grey")

    def test_dull_green_paint_is_green(self):
        # Lime/olive paint in shade: well below blue's S >= 100 bar.
        self.assertEqual(hsv_colour(solid_hsv_image(40, 80, 150)), "green")

    def test_white_in_warm_light_is_not_brown(self):
        # Beige-looking pixels, but the whole scene is just as warm.
        cream = solid_hsv_image(20, 45, 200)
        self.assertEqual(hsv_colour(cream, scene_warm=0.05), "brown")
        self.assertNotEqual(hsv_colour(cream, scene_warm=0.9), "brown")

    def test_dull_beige_paint_is_brown(self):
        self.assertEqual(hsv_colour(solid_hsv_image(20, 37, 180)), "brown")

    def test_mask_ignores_neighbouring_car_inside_box(self):
        # Left 60% is this white car, right 40% is a red neighbour that the
        # box also covers. Without the mask the red neighbour wins.
        hsv = np.full((100, 100, 3), (0, 0, 220), dtype=np.uint8)
        hsv[:, 60:] = (0, 200, 170)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, :60] = True

        self.assertEqual(hsv_colour(rgb), "red")
        self.assertEqual(hsv_colour(rgb, mask), "white")

    def test_tiny_mask_falls_back_to_body_band(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:3, 0:3] = True
        self.assertEqual(hsv_colour(solid_hsv_image(0, 0, 240), mask), "white")


if __name__ == "__main__":
    unittest.main()
