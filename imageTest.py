import os

import cv2

from libs.bColor import bcolors
from libs.testGen import generateTestSquare
from libs.imageFusion import TransparentImageOverlay


# set image creation runs
count = 50

# Create test folder
myPath = f"output/_test/run-X{count}-V"

prefixNbr = 1
folder = myPath+f"{prefixNbr}/"

while True:
    if not os.path.isdir(folder):
        print("folder doesn't exist", folder)
        os.makedirs(folder)
        break

    # check if folder maybe empty
    if len(os.listdir(folder)) == 0:
        print("folder is empty -> use it", folder)
        break

    prefixNbr = prefixNbr + 1
    folder = myPath+f"{prefixNbr}/"

print("created folder", folder)

# generate random transformed rectangles for test purposes
print(f"Generating {count} images...")
testEngine = generateTestSquare(count, folder)
images = testEngine.test_run()

# initialize testing class
imageAnalyzer = TransparentImageOverlay("", "")
print("...successful")

blue_c = (255,0,0)
red_c = (0,0,255)

# Create Demo Image Test method with image Analyzer
def hughPointLinecheck(contour, background):
    cv2.imwrite(folder + f"{x}-1-canny.png", contour)

    lines = imageAnalyzer.get_hough_lines(contour)
    mapped_lines_x, mapped_lines_y = imageAnalyzer.combine_overlapping_lines(lines)

    mapped_background = background.copy()
    final_border = background.copy()

    def print_lines(lines_data, image, color=blue_c):
        if lines_data is not None:
            for lin in lines_data:
                x1, y1, x2, y2 = lin[0]
                cv2.line(image, (x1, y1), (x2, y2), color, 3)
                cv2.circle(image, (x1, y1), 15, color, 3)
                cv2.circle(image, (x2, y2), 15, color, 3)

    def print_dots(dot_data, image, color=blue_c):
        if dot_data is not None:
            for dot in dot_data:
                x, y, = dot
                cv2.circle(image, (x, y), 15, color, 3)

    if lines is not None:
        print(f"PointCheck: found {len(lines)} lines in Image")
        print_lines(lines, background)
    else:
        print("PointCheck: No contour was found on the image")
        return False

    cv2.imwrite(folder + f"{x}-2-coord.png", background)

    if mapped_lines_x is not None:
        print(f"PointCheck: mapped to {len(mapped_lines_x)} horizontal lines in Image")
        print_lines(mapped_lines_x, mapped_background)
    else:
        return False

    if mapped_lines_y is not None:
        print(f"PointCheck: mapped to {len(mapped_lines_y)} vertical lines in Image")
        print_lines(mapped_lines_y, mapped_background, red_c)
    else:
        return False

    intersect_color = (255, 255, 0)
    # Find relevant lines from vertical/horizontal mapped
    # Sort Lines by Central Cross-Section
    sorted_y_intersection, vert_line = imageAnalyzer.get_relevant_lines(mapped_lines_x, True)
    if len(sorted_y_intersection) < 2:
        print(f"{bcolors.FAIL}Could not find any horizontal mapped lines -> abort{bcolors.ENDC}")
        return False
    y_points, y_lines = zip(*sorted_y_intersection)

    # print intersections with vertical reference line on mapped_background
    print_dots(y_points, mapped_background, intersect_color)
    print_lines([(vert_line,)], mapped_background, intersect_color)

    sorted_x_intersection, hori_line = imageAnalyzer.get_relevant_lines(mapped_lines_y, False)
    if len(sorted_x_intersection) < 2:
        print(f"{bcolors.FAIL}Could not find any vertical mapped lines -> abort{bcolors.ENDC}")
        return False
    x_points, x_lines = zip(*sorted_x_intersection)

    # print intersections with horizontal reference line on mapped_background
    print_dots(x_points, mapped_background, intersect_color)
    print_lines([(hori_line,)], mapped_background, intersect_color)

    cv2.imwrite(folder + f"{x}-3-mapped.png", mapped_background)

    four_borders = [y_lines[0], x_lines[-1], y_lines[-1], x_lines[0]]
    # print 4-found lines - with Intersection points
    print_lines(four_borders, final_border)
    cv2.imwrite(folder + f"{x}-4-reduced.png", final_border)

    transform_points = imageAnalyzer.get_screen_position(four_borders)

    if transform_points is not None:
        print_dots(transform_points, final_border, (255,255,255))
        cv2.imwrite(folder + f"{x}-4-reduced.png", final_border)
    else:
        print(f"{bcolors.FAIL}Could not find any intersections{bcolors.ENDC}")
        return False

    print("PointCheck: Saved report image")
    return True


successCount = 0
for (x,data) in enumerate(images):
    # try to find solution
    print(f"\nAnalyze image {x + 1}/{count}")

    canny = data.get("canny")
    bg = data.get("original")

    if hughPointLinecheck(canny, bg):
        successCount += 1

print(f"\n\nAnalysed with Rate of {successCount} / {count}")
