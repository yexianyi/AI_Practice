import pandas as pd

# 读取历史数据
data = pd.read_csv("daletou.csv")

def get_last_n_records(data, n):
    """获取最近的n条记录"""
    return data.tail(n)

def max_minus_min(last_n_records):
    """前区最大四码减最小号法"""
    last_record = last_n_records.iloc[-1]
    numbers = [last_record['R1'], last_record['R2'], last_record['R3'], last_record['R4'], last_record['R5']]
    numbers.sort()
    max_four = numbers[-4:]
    min_number = numbers[0]
    exclusion_numbers = [num - min_number for num in max_four]
    return exclusion_numbers

def add_three_kill_tail(last_n_records):
    """前区号码加3杀尾法"""
    last_record = last_n_records.iloc[-1]
    numbers = [last_record['R1'], last_record['R2'], last_record['R3'], last_record['R4'], last_record['R5']]
    exclusion_tails = set()
    result = numbers[0] + 3 + numbers[1] + 3 + numbers[2] + 3 + numbers[3] + 3 + numbers[4] + 3
    if result > 35:
        for num in numbers:
            tail = (num + 3) % 10
            for red_candidate in range(1, 36):  # 1到35
                if red_candidate % 10 == tail:
                    exclusion_tails.add(red_candidate)   # 处理尾数和号码对应关系s

    return list(exclusion_tails)

def two_period_diff(last_n_records):
    """两期前区前三码互减法"""
    if len(last_n_records) < 2:
        return []
    prev_record = last_n_records.iloc[-2]
    last_record = last_n_records.iloc[-1]
    prev_numbers = [prev_record['R1'], prev_record['R2'], prev_record['R3']]
    last_numbers = [last_record['R1'], last_record['R2'], last_record['R3']]
    exclusion_numbers = set()
    for i in range(3):
        diff = abs(prev_numbers[i] - last_numbers[i])
        exclusion_numbers.add(diff)
    return list(exclusion_numbers)

def max_span_kill(last_n_records):
    """前区最大跨度值杀号法"""
    last_record = last_n_records.iloc[-1]
    numbers = [last_record['R1'], last_record['R2'], last_record['R3'], last_record['R4'], last_record['R5']]
    span = max(numbers) - min(numbers)
    return [span]

def back_area_kill(last_n_records):
    """后区和值杀号法"""
    last_record = last_n_records.iloc[-1]
    sum_back = last_record['B1'] + last_record['B2']
    if sum_back > 35:
        return [sum_back - 35]
    return [sum_back]

def overlap_kill_for_blue(last_n_records):
    """前后区重叠号码杀号法"""
    last_record = last_n_records.iloc[-1]
    front_numbers = {last_record['R1'], last_record['R2'], last_record['R3'], last_record['R4'], last_record['R5']}
    back_numbers = {last_record['B1'], last_record['B2']}
    overlap = front_numbers & back_numbers  # 交集
    return list(overlap)

def check_strategy_accuracy(func, color):
    correct_case_counter = 0
    total_case_counter = 0
    for i in range(1, len(data)-2):
        rows = data.iloc[i: i+1]
        predict_exclusion_numbers = func(rows)
        actual_numbers = data.iloc[i+1: i+2]
        actual_red_numbers = {actual_numbers['R1'].values[0], actual_numbers['R2'].values[0], actual_numbers['R3'].values[0], actual_numbers['R4'].values[0], actual_numbers['R5'].values[0]}
        actual_blue_numbers = {actual_numbers['B1'].values[0], actual_numbers['B2'].values[0]}
        overlaps_nums = ()
        if color == 'RED':
            overlaps_nums = actual_red_numbers & set(predict_exclusion_numbers)
        if color == 'BLUE':
            overlaps_nums = actual_blue_numbers & set(predict_exclusion_numbers)

        if len(overlaps_nums) == 0:
            correct_case_counter += 1
        total_case_counter += 1

    accuracy = (correct_case_counter / total_case_counter) * 100
    return f"{accuracy:.{2}f}%"




def main():
    last_n_records = get_last_n_records(data, n=2)  # 根据需要调整获取的记录数
    print("红球杀号：")
    print("准确率:", check_strategy_accuracy(max_minus_min, 'RED'), "前区最大四码减最小号法", max_minus_min(last_n_records))
    print("准确率:", check_strategy_accuracy(add_three_kill_tail, 'RED'), "前区号码加3杀尾法:", add_three_kill_tail(last_n_records))
    print("准确率:", check_strategy_accuracy(two_period_diff, 'RED'), "两期前区前三码互减法:", two_period_diff(last_n_records))
    print("准确率:", check_strategy_accuracy(max_span_kill, 'RED'), "前区最大跨度值杀号法:", max_span_kill(last_n_records))
    print("准确率:", check_strategy_accuracy(back_area_kill, 'RED'), "后区和值杀号法:", back_area_kill(last_n_records))

    print("蓝球杀号：")
    print("准确率:", check_strategy_accuracy(back_area_kill, 'BLUE'), "前后区重叠号码杀号法:", overlap_kill_for_blue(last_n_records))

if __name__ == "__main__":
    main()