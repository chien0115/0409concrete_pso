function [E, all_dispatch_times] = objective_function(P, t, time_windows, num_sites, dispatch_times, work_time, time, max_interrupt_time, penalty)
[x1, y1] = size(P);  % 獲取染色體數量和每個染色體的位元數
num_dispatch_order = y1;  % 派遣順序的大小
H = zeros(1, x1);  % 初始化適應度值
all_dispatch_times = zeros(x1, t);

% Initialize tables for waiting times
truck_waiting_times = zeros(x1, num_dispatch_order);
site_waiting_times = zeros(x1, num_dispatch_order);
all_work_start_times = zeros(x1, num_dispatch_order);

for i = 1:x1  % 遍歷每個染色體
    penalty_side_time = 0;  % 每個工地的懲罰時間
    penalty_truck_time = 0;  % 卡車等待懲罰時間
    dispatch_times_for_chromosome = dispatch_times(i, :);  % 每行染色體的派遣時間(t個)
    dispatch_order_for_chromosome = P(i, :);  % 每個染色體派遣順序
    actual_dispatch_time = zeros(1, num_dispatch_order);  % 实际派遣时间
    travel_to_site = zeros(1, num_dispatch_order);  % 每個到的時間
    travel_back_site = zeros(1, num_dispatch_order);
    arrival_times = zeros(1, num_dispatch_order);
    site_set_start_times = zeros(1, num_dispatch_order);
    work_start_times = zeros(1, num_dispatch_order);  % 每個開始時間
    finish_time_site = zeros(1, num_dispatch_order);  % 每個結束時間
    return_times = zeros(1, num_dispatch_order);
    truck_availability = zeros(1, t);  % 追踪每台卡車何時可以再次使用

    for k = 1:num_dispatch_order
        site_id = dispatch_order_for_chromosome(k);
        travel_to_site(k) = time(site_id, 1);
        travel_back_site(k) = time(site_id, 2);
        site_set_start_times(k) = time_windows(site_id, 1);

        % 設計工地派遣時間 開始被派遣
        if k <= t
            % 前 t 台車直接派遣
            truck_id = k;
            actual_dispatch_time(k) = dispatch_times_for_chromosome(k);
            all_dispatch_times(i, k) = actual_dispatch_time(k);  % 只記錄前 t 台車的派遣時間
        else
            % 找出最早可用的車輛
            [next_available_time, truck_id] = min(truck_availability(1:t));  % 只考慮前 t 台車
            actual_dispatch_time(k) = next_available_time;  % 用最早可用的時間作為派遣時間
        end

        arrival_times(k) = actual_dispatch_time(k) + travel_to_site(k);

        % 檢查之前是否有車在該工地工作
        previous_work_idx = find(dispatch_order_for_chromosome(1:k-1) == site_id, 1, 'last');

        if isempty(previous_work_idx)
            % 如果這是該工地的第一台車
            if arrival_times(k) < time_windows(site_id, 1)
                % 如果到達時間早於工地開始時間，要等到工地開始時間
                work_start_times(k) = time_windows(site_id, 1);
            else
                % 如果到達時間晚於工地開始時間，直接開始工作
                work_start_times(k) = arrival_times(k);
            end
        else
            % 如果不是第一台車，需要考慮前一台車的完成時間
            if arrival_times(k) < finish_time_site(previous_work_idx)
                % 如果到達時間早於前一台車完成時間，需要等待
                work_start_times(k) = finish_time_site(previous_work_idx);
            else
                % 如果到達時間晚於前一台車完成時間
                work_start_times(k) = max(arrival_times(k), time_windows(site_id, 1));
            end
        end


        % 儲存工作開始時間
        all_work_start_times(i, k) = work_start_times(k);


        % 計算完成時間和返回時間
        finish_time_site(k) = work_start_times(k) + work_time(site_id);
        return_times(k) = finish_time_site(k) + travel_back_site(k);
        truck_availability(truck_id) = return_times(k);  % 更新該台車的可用時間

        % 计算等待时间
        if ~isempty(previous_work_idx)
            % 如果之前有车已经到过该工地，判断是卡车等待还是工地等待
            if arrival_times(k) < finish_time_site(previous_work_idx)
                truck_waiting_times(i, k) = finish_time_site(previous_work_idx) - arrival_times(k);
                penalty_truck_time = penalty_truck_time + truck_waiting_times(i, k);
            elseif arrival_times(k) > finish_time_site(previous_work_idx)
                site_waiting_times(i, k) = arrival_times(k) - finish_time_site(previous_work_idx);
                if site_waiting_times(i, k) > max_interrupt_time(site_id)
                    penalty_side_time = penalty_side_time + 1;  % 增加懲罰
                end
            end
        else
            % 第一次到该工地
            if arrival_times(k) < site_set_start_times(k)
                truck_waiting_times(i, k) = site_set_start_times(k) - arrival_times(k);
                penalty_truck_time = penalty_truck_time + truck_waiting_times(i, k);
            else
                site_waiting_times(i, k) = arrival_times(k) - site_set_start_times(k);
                if site_waiting_times(i, k) > max_interrupt_time(site_id)
                    penalty_side_time = penalty_side_time + 1;  % 增加懲罰
                end
            end
        end

    end

    total_penalty = penalty_side_time * penalty + penalty_truck_time;  % 總懲罰值
    H(i) = total_penalty;  % 適應度值
end

E = H;  % Return fitness values

% Create a combined waiting time and dispatch time table
% waiting_time_table = table((1:x1)', ...
%     sum(truck_waiting_times, 2), ...  % Sum the truck waiting times for each chromosome
%     sum(site_waiting_times, 2), ...   % Sum the site waiting times for each chromosome
%     all_dispatch_times, ...
%     all_work_start_times, ...         % 使用正確保存的工作開始時間
%     truck_waiting_times, ...          % Add individual truck waiting times
%     site_waiting_times, ...           % Add individual site waiting times
%     'VariableNames', {'Chromosome_ID', 'Total_Truck_Waiting_Times', 'Total_Site_Waiting_Times', ...
%                      'Dispatch_Times', 'Work_Start_Times', 'Truck_Waiting_Times', 'Site_Waiting_Times'});
%
% % Display the waiting time table
% disp(waiting_time_table);  % Display the table
end

