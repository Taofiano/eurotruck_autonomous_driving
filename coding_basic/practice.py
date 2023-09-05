# import numpy as np
#
# # lines = [[1, 2, 3, 4],
# #          [1, 2, 1, 7],
# #          [1, 2, -2, 5],
# #          [1, 2, -5, 2]]
#
# # lines = np.array(lines)
# lines = np.array([[826, 434, 1164, 688],
#  [822, 451, 1087, 698],
#  [ 57, 718, 282, 588],
#  [926, 508, 1207, 719],
#  [892, 515, 1103, 712],
#  [29, 710, 249, 589],
#  [572, 669, 621, 542],
#  [802, 433, 926, 549],
#  [ 10, 719, 110, 665],
#  [ 85, 678, 258, 586],
#  [823, 432, 937, 518],
#  [ 76, 706, 280, 588]])
#
# theta_deg = np.array([36.92410367, 42.9865245, -30.01836743, 36.90250765, 43.03473943, -28.81079374, -68.90205041, 43.09084757, -28.36904629, -28.00368387, 37.03038961, -30.04643525])
#
# line_pos_arr = np.empty((0, 4), float)
# line_neg_arr = np.empty((0, 4), float)
#
# for index in range(theta_deg.size):
#     if index < (theta_deg.size - 1):
#         if theta_deg[index] > 0:
#             if abs(theta_deg[index] - theta_deg[index + 1]) < 10:
#                 np.append(line_pos_arr, lines[index], axis=0)
#
#         elif theta_deg[index] < 0:
#             if abs(theta_deg[index] - theta_deg[index + 1]) < 10:
#                 np.append(line_neg_arr, lines[index], axis=0)
#
# print(line_pos_arr)
# print(line_neg_arr)

import numpy as np

# lines = [[1, 2, 3, 4],
#          [1, 2, 1, 7],
#          [1, 2, -2, 5],
#          [1, 2, -5, 2]]

# lines = np.array(lines)
lines = np.array([[826, 434, 1164, 688],
 [822, 451, 1087, 698],
 [ 57, 718, 282, 588],
 [926, 508, 1207, 719],
 [892, 515, 1103, 712],
 [29, 710, 249, 589],
 [572, 669, 621, 542],
 [802, 433, 926, 549],
 [ 10, 719, 110, 665],
 [ 85, 678, 258, 586],
 [823, 432, 937, 518],
 [ 76, 706, 280, 588]])

theta_deg = np.array([36.92410367, 42.9865245, -30.01836743, 36.90250765, 43.03473943, -28.81079374, -68.90205041, 43.09084757, -28.36904629, -28.00368387, 37.03038961, -30.04643525])

line_pos_arr = np.empty((0, 4), float)
line_neg_arr = np.empty((0, 4), float)

for index in range(theta_deg.size):
    if index < (theta_deg.size - 1):
        if theta_deg[index] > 0:
            if abs(theta_deg[index] - theta_deg[index + 1]) < 10:
                line_pos_arr = np.vstack((line_pos_arr, lines[index]))

        elif theta_deg[index] < 0:
            if abs(theta_deg[index] - theta_deg[index + 1]) < 10:
                line_neg_arr = np.vstack((line_neg_arr, lines[index]))

print(line_pos_arr)
print(line_neg_arr)
