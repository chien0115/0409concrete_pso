clc;
clear all;
close all;

% Problem Definition
t = 5; % 卡車數
num_sites = 3; % 工地
time_windows = [480, 1440;
    480, 1440;
    510, 1440;
    ]; % 每個工地的時間窗

time = [30, 25;  % 去程到工地 1 需要 30 分鐘，回程需要 25 分鐘
    25, 20;
    40, 30;
    ];

max_interrupt_time = [30,20,15]; % 工地最大容許中斷時間 (分鐘)
work_time = [20,30,25]; % 各工地施工時間 (分鐘)
demand_trips = [3,4,5]; % 各工地需求車次
penalty = 24*60; % 懲罰值
total_trips = sum(demand_trips);

% PSO Parameters
MaxIter = 300;
nPop = 200;
w = 0.9; % 慣性權重(控制粒子保持原本移動方向的趨勢)
wdamp = 0.99; % 慣性權重衰減率
c1 = 2; % 個人學習係數(控制向個體最佳解學習的程度)
c2 = 2; % 群體學習係數(控制向群體最佳解學習的程度)

% 粒子結構體
empty_particle.Position = [];            % 粒子位置(派遣順序)
empty_particle.Velocity = [];            % 粒子速度
empty_particle.Cost = [];                % 成本值
empty_particle.RealFitness = [];         % 實際適應度
empty_particle.DispatchTimes = [];       % 派遣時間
empty_particle.Best.Position = [];       % 個體最佳位置 
% 在複製後，particle(1), particle(2)...每個粒子都會有自己的 Best.Position 似pbest
empty_particle.Best.Cost = inf;          % 個體最佳成本
empty_particle.Best.RealFitness = -inf;  % 個體最佳適應度
empty_particle.Best.DispatchTimes = [];  % 個體最佳派遣時間

particle = repmat(empty_particle, nPop, 1);  % 複製nPop個粒子   nPop: 要複製的行數(粒子數量) 1: 列數(這裡是一維陣列)
GlobalBest.Cost = inf;                       % 初始化全域最佳成本
GlobalBest.RealFitness = -inf;              % 初始化全域最佳適應度
GlobalBest.Position = [];                    % 初始化全域最佳位置
GlobalBest.DispatchTimes = [];              % 初始化全域最佳派遣時間

max_temp_fitness = -inf;
K = zeros(MaxIter, 2);

% Initialize particles
for i = 1:nPop
    % Create dispatch order
    dispatch_order = [];
    random_values = [];
    for j = 1:num_sites
        dispatch_order = [dispatch_order, repmat(j, 1, demand_trips(j))];%直的
        random_values = [random_values, rand(1, demand_trips(j))]; %產生demand_trips(j)組0~1數字
    end
    dispatch_random_pairs = [dispatch_order; random_values]';
    sorted_pairs = sortrows(dispatch_random_pairs, 2);
    sorted_dispatch_order = sorted_pairs(:, 1)';
    


    % Set particle position and velocity
    particle(i).Position = sorted_dispatch_order;
    % % 加入修復
    particle(i).Position = repair_position(particle(i).Position, demand_trips);
    particle(i).Velocity = zeros(1, total_trips);


    % Initialize dispatch times
    particle(i).DispatchTimes = zeros(1, total_trips);
    for k = 1:total_trips
        site_idx = particle(i).Position(k);
        early_time = time_windows(site_idx, 1);
        time_range = time_windows(site_idx, 2) - time_windows(site_idx, 1);
        random_time = rand()^3;
        particle(i).DispatchTimes(k) = round(early_time + random_time * time_range - time(site_idx, 1));
    end

    
    % Calculate initial cost
    [particle(i).Cost, actual_dispatch_times] = objective_function(particle(i).Position, ...
        t, time_windows, num_sites, particle(i).DispatchTimes, ...
        work_time, time, max_interrupt_time, penalty);
    particle(i).RealFitness = -particle(i).Cost; %加負號的原因是為了將最小化問題轉換為最大化問題。負最小最好
    particle(i).ActualDispatchTimes = actual_dispatch_times(1,:); % 保存實際派遣時間
    
    % Initialize personal best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.DispatchTimes = particle(i).DispatchTimes;
    particle(i).Best.ActualDispatchTimes = particle(i).ActualDispatchTimes; % 保存到個體最佳解
    particle(i).Best.Cost = particle(i).Cost;
    particle(i).Best.RealFitness = particle(i).RealFitness;
    
    % Update global best if needed
    if particle(i).Best.RealFitness > GlobalBest.RealFitness
        GlobalBest = particle(i).Best;
    end
end


% Main PSO Loop
BestCosts = zeros(MaxIter, 1);
figure;
hold on;
grid on;
title('PSO Blue-Average      Red-Minimum');
xlabel('Generation')
ylabel('Objective Function Value')



%讓粒子根據當前的個體最佳(pBest)和全域最佳(gBest)進行移動
for it = 1:MaxIter
    for i = 1:nPop
        % Update Velocity and Position for dispatch order
        r1 = rand(1, total_trips);
        r2 = rand(1, total_trips);
        particle(i).Velocity = w * particle(i).Velocity + ...
            c1 * r1 .* (particle(i).Best.Position - particle(i).Position) + ...
            c2 * r2 .* (GlobalBest.Position - particle(i).Position);
        %第一次迭代時, w * particle(i).Velocity 確實會是 0
        new_position = particle(i).Position + particle(i).Velocity;
        
        % Ensure position values stay within site bounds and repair
        %因new_position出來有可能是有小數的，所以要使用round 而min和max是要限制產生的工地都在有的工地編號內
        preliminary_position = max(1, min(num_sites, round(new_position)));
        particle(i).Position = repair_position(preliminary_position, demand_trips);
        
        % Update dispatch times
        for j = 1:total_trips
            site_idx = particle(i).Position(j);
            early_time = time_windows(site_idx, 1);
            late_time = time_windows(site_idx, 2);
            %注意
            current_time = particle(i).DispatchTimes(j);
            pbest_time = particle(i).Best.DispatchTimes(j);
            gbest_time = GlobalBest.DispatchTimes(j);
            
            new_time = w * current_time + ...
                c1 * rand() * (pbest_time - current_time) + ...
                c2 * rand() * (gbest_time - current_time);
            
            %max(early_time - time(site_idx, 1)確保不早於最早時間
            %min(late_time - time(site_idx, 1)確保不超過最晚時間
            particle(i).DispatchTimes(j) = max(early_time - time(site_idx, 1), ...
                min(late_time - time(site_idx, 1), round(new_time)));
        end
        
        [particle(i).Cost, actual_dispatch_times] = objective_function(particle(i).Position, ...
            t, time_windows, num_sites, particle(i).DispatchTimes, ...
            work_time, time, max_interrupt_time, penalty);
        particle(i).RealFitness = -particle(i).Cost;
        particle(i).ActualDispatchTimes = actual_dispatch_times(1,:); % 保存實際派遣時間
        
        % Update personal best
        if particle(i).RealFitness > particle(i).Best.RealFitness
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.DispatchTimes = particle(i).DispatchTimes;
            particle(i).Best.ActualDispatchTimes = particle(i).ActualDispatchTimes; % 更新個體最佳解的實際派遣時間
            particle(i).Best.Cost = particle(i).Cost;
            particle(i).Best.RealFitness = particle(i).RealFitness;
            
            % Update global best
            if particle(i).Best.RealFitness > GlobalBest.RealFitness
                GlobalBest = particle(i).Best;
            end
        end
    end
    
    % Update inertia weight
    w = w * wdamp;
    
    % Store best cost
    BestCosts(it) = -GlobalBest.Cost;
    
    % Calculate and store statistics
    costs = [particle.Cost];
    K(it, 1) = mean(costs);  % Average fitness
    K(it, 2) = min(costs);   % Best fitness

    %這段程式碼是在記錄每一代的最佳值，使用了不同的策略來處理第一代和之後的世代
    % 修改最佳值的記錄方式
    % if it == 1%代表第一世代
    %     K_PSO(it, 2) = min(costs);  % 第一代直接記錄最佳值
    % else
    %     K_PSO(it, 2) = min(K_PSO(it-1, 2), min(costs));  % 與前一代比較，保留較好的
    % end

    %K_PSO(1, 2) = 80  上一代記錄的最佳值K_PSO(2, 2) = min(80, 75) = 75  % 比較並保留較好的值
    
    % Update plot
    plot(it, K(it, 1), 'b.');
    plot(it, K(it, 2), 'r.');
    drawnow;
    
    % Display iteration info
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(-GlobalBest.Cost)]);
end

% Get best solution
best_chromosome = GlobalBest.Position;
best_dispatch_times = GlobalBest.DispatchTimes;

disp('Best Chromosome:');
disp(best_chromosome);
disp('Best Dispatch Times (Planned):');
disp(best_dispatch_times);
disp('Best Dispatch Times (Actual):');
disp(GlobalBest.ActualDispatchTimes);

best_chromosome_evaluation = objective_function(best_chromosome, t, time_windows, num_sites,best_dispatch_times, work_time, time, max_interrupt_time, penalty);
disp('Best Evaluation:');
disp(best_chromosome_evaluation);


% Decode solution
%解碼最佳解為派車計劃
dispatch_plan = decode_chromosome(best_chromosome, best_dispatch_times, t, demand_trips, time_windows, work_time, time);

% 展示派車順序表
vehicle_ids = dispatch_plan(:, 1);
site_ids = dispatch_plan(:, 2);
actual_dispatch_times = dispatch_plan(:, 3);
travel_times_to = dispatch_plan(:, 4);
arrival_times = dispatch_plan(:, 5);
site_set_start_times = dispatch_plan(:, 6); % 新增的工作開始時間
work_start_times = dispatch_plan(:, 7); % 新增的工作開始時間
work_times = dispatch_plan(:, 8);
site_finish_times = dispatch_plan(:, 9);
travel_times_back = dispatch_plan(:, 10);
return_times = dispatch_plan(:, 11);
truck_waiting_times = dispatch_plan(:, 12);
site_waiting_times = dispatch_plan(:, 13);





% 將時間轉換為 HH:MM 格式
actual_dispatch_times_formatted = cellstr(arrayfun(@convert_minutes_to_time, actual_dispatch_times, 'UniformOutput', false));
arrival_times_formatted = cellstr(arrayfun(@convert_minutes_to_time, arrival_times, 'UniformOutput', false));
return_times_formatted = cellstr(arrayfun(@convert_minutes_to_time, return_times, 'UniformOutput', false));
site_finish_times_formatted = cellstr(arrayfun(@convert_minutes_to_time, site_finish_times, 'UniformOutput', false));
truck_waiting_times_formatted = cellstr(arrayfun(@(x) sprintf('%d min', x), truck_waiting_times, 'UniformOutput', false));
site_waiting_times_formatted = cellstr(arrayfun(@(x) sprintf('%d min', x), site_waiting_times, 'UniformOutput', false));
site_set_start_times_formatted = cellstr(arrayfun(@convert_minutes_to_time, site_set_start_times, 'UniformOutput', false));
work_start_times_formatted = cellstr(arrayfun(@convert_minutes_to_time, work_start_times, 'UniformOutput', false)); % 新增工作開始時間格式化

% 創建派遣計劃表格數據
dispatch_data = table(vehicle_ids, site_ids, actual_dispatch_times_formatted, travel_times_to, arrival_times_formatted, site_set_start_times_formatted, work_start_times_formatted, work_times, site_finish_times_formatted, travel_times_back, return_times_formatted, truck_waiting_times_formatted, site_waiting_times_formatted, ...
    'VariableNames', {'VehicleID', 'SiteID', 'ActualDispatchTime', 'TravelTimeTo', 'ArrivalTime', 'SiteSetTime', 'WorkStartTime', 'WorkTime', 'SiteFinishTime', 'TravelTimeBack', 'ReturnTime', 'TruckWaitingTime', 'SiteWaitingTime'});



% 顯示表格
figure;
uitable('Data', table2cell(dispatch_data), 'ColumnName', dispatch_data.Properties.VariableNames, ...
    'RowName', [], 'Position', [20 20 800 400]);

% 解码函数
% 修改 decode_chromosome 函數的開頭部分
% 修改 decode_chromosome 函數的開頭部分
function plan = decode_chromosome(chromosome, dispatch_times, t, demand_trips, time_windows, work_time, time)
    fprintf('Chromosome:\n');
    disp(chromosome);
    
    total_trips = sum(demand_trips);
    site_ids = zeros(total_trips, 1);
    actual_dispatch_times = zeros(total_trips, 1);
    travel_times_to = zeros(total_trips, 1);
    arrival_times = zeros(total_trips, 1);
    site_set_start_times = zeros(total_trips, 1);
    work_start_times = zeros(total_trips, 1);
    work_times = zeros(total_trips, 1);
    site_finish_times = zeros(total_trips, 1);
    travel_times_back = zeros(total_trips, 1);
    return_times = zeros(total_trips, 1);
    truck_waiting_times = zeros(total_trips, 1);
    site_waiting_times = zeros(total_trips, 1);

    % 初始化卡車可用時間
    truck_availability = zeros(t, 1);

    for i = 1:total_trips
        site_ids(i) = chromosome(1,i);
        site_id = site_ids(i);

        % % 使用傳入的 dispatch_times，而不是重新計算
        % actual_dispatch_times(i) = dispatch_times(i);

        % 獲取各個時間參數
        travel_times_to(i) = time(site_id,1);
        travel_times_back(i) = time(site_id,2);
        site_set_start_times(i) = time_windows(site_id,1);
        work_times(i) = work_time(site_id);

        % 設計工地派遣時間
        if i <= t
            % 前 t 台車使用給定的派遣時間
            actual_dispatch_times(i) = dispatch_times(i);
            truck_id = i;
        else
            % t 台車之後，找出最早可用的車輛
            [next_available_time, truck_id] = min(truck_availability);
            actual_dispatch_times(i) = next_available_time;  % 使用最早可用時間作為派遣時間
        end

        % 計算到達時間
        arrival_times(i) = actual_dispatch_times(i) + travel_times_to(i);

        % 檢查之前是否有卡車在該工地工作
        previous_work_idx = find(site_ids(1:i-1) == site_ids(i), 1, 'last');
        if isempty(previous_work_idx)
            work_start_times(i) = max(arrival_times(i), site_set_start_times(i));
        else
            work_start_times(i) = max(arrival_times(i), site_finish_times(previous_work_idx));
        end

        % 計算工地完成時間和卡車返回時間
        site_finish_times(i) = work_start_times(i) + work_times(i);
        return_times(i) = site_finish_times(i) + travel_times_back(i);
        truck_availability(truck_id) = return_times(i);  % 更新該台車的可用時間

        % 計算等待時間
        if ~isempty(previous_work_idx)
            if arrival_times(i) < site_finish_times(previous_work_idx)
                truck_waiting_times(i) = site_finish_times(previous_work_idx) - arrival_times(i);
            elseif arrival_times(i) > site_finish_times(previous_work_idx)
                site_waiting_times(i) = arrival_times(i) - site_finish_times(previous_work_idx);
            end
        else
            if arrival_times(i) < site_set_start_times(i)
                truck_waiting_times(i) = site_set_start_times(i) - arrival_times(i);
            else
                site_waiting_times(i) = arrival_times(i) - site_set_start_times(i);
            end
        end

        % 打印調試信息
        fprintf('Trip %d: Site %d, Dispatch: %f, Arrival: %f, Work Start: %f, Return: %f\n', ...
            i, site_ids(i), actual_dispatch_times(i), arrival_times(i), work_start_times(i), return_times(i));
    end

    vehicle_ids = (1:total_trips)';
    plan = [vehicle_ids, site_ids, actual_dispatch_times, travel_times_to, arrival_times, site_set_start_times, work_start_times, work_times, site_finish_times, travel_times_back, return_times, truck_waiting_times, site_waiting_times];
end



% 分鐘轉換為 HH:MM 格式的函數
function time_str = convert_minutes_to_time(minutes)
hours = floor(minutes / 60);
mins = mod(minutes, 60);
time_str = sprintf('%02d:%02d', hours, mins);
end