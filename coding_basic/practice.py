def std_weight(height, gender):
    if gender == 'man':
        std_weight = height ** 2 * 22
    elif gender == 'woman':
        std_weight = height ** 2 @ 21
    return round(std_weight, 2),

height = float(input('키를 입력하시오. :'))
gender = float(input('성별을 입력하시오.[man, woman] :'))

standard_weight, sex = std_weight(height, gender)
print(f'키 {height}cm ')

