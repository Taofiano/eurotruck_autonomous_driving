import numpy as np

lines = np.array([[826, 434, 1164, 688],
                  [822, 451, 1087, 698],
                  [57, 718, 282, 588],
                  [926, 508, 1207, 719],
                  [892, 515, 1103, 712],
                  [29, 710, 249, 589],
                  [572, 669, 621, 542],
                  [802, 433, 926, 549],
                  [10, 719, 110, 665],
                  [85, 678, 258, 586],
                  [823, 432, 937, 518],
                  [76, 706, 280, 588]])

theta_deg = np.array([36.92410367, 42.9865245, -30.01836743, 36.90250765, 43.03473943, -28.81079374, -68.90205041, 43.09084757, -28.36904629, -28.00368387, 37.03038961, -30.04643525])

line_pos_arr = np.empty((0, 4), float)
line_neg_arr = np.empty((0, 4), float)

deg_pos_mean = 0
deg_neg_mean = 0

deg_pos_count = 0
deg_neg_count = 0

for index in range(theta_deg.size):
    if index < (theta_deg.size - 1):
        if theta_deg[index] > 0:
            line_pos_arr = np.vstack((line_pos_arr, lines[index]))
            deg_pos_count += 1
            if deg_pos_count == 1:
                deg_pos_mean += theta_deg[index]
            else:
                deg_pos_mean = (deg_pos_mean + theta_deg[index]) / 2

            if abs(deg_pos_mean - theta_deg[index + 1]) < 10:
                line_pos_arr = np.vstack((line_pos_arr, lines[index + 1]))
            # else:
            #     line_pos_arr = np.vstack((line_pos_arr, lines[index]))

        elif theta_deg[index] < 0:
            line_neg_arr = np.vstack((line_neg_arr, lines[index]))
            deg_neg_count += 1
            if deg_neg_count == 1:
                deg_neg_mean += theta_deg[index]
            else:
                deg_neg_mean = (deg_neg_mean + theta_deg[index]) / 2
            if abs(deg_neg_mean - theta_deg[index + 1]) < 10:
                line_neg_arr = np.vstack((line_neg_arr, lines[index + 1]))
            # else:
            #     line_neg_arr = np.vstack((line_neg_arr, lines[index]))
    elif index == theta_deg.size:
        index = -1

line_pos_arr = np.unique(line_pos_arr, axis=0)
line_neg_arr = np.unique(line_neg_arr, axis=0)

print(line_pos_arr)
print(line_neg_arr)
