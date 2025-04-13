function repaired_position = repair_position(position, demand_trips)
num_sites = length(demand_trips);
site_counts = zeros(1, num_sites);

% 計算當前訪問次數
for i = 1:num_sites
    site_counts(i) = sum(position == i);
end

% 修復過多派遣次數的工地
for site = 1:num_sites
    diff = site_counts(site) - demand_trips(site);

    while diff > 0
        % 找到需求不足的工地（按差異排序）
        shortages = demand_trips - site_counts;
        [sorted_shortages, shortage_idx] = sort(shortages, 'descend');
        under_demand_sites = shortage_idx(sorted_shortages > 0);

        if isempty(under_demand_sites)
            break;
        end

        % 選擇最需要補充的工地
        new_site = under_demand_sites(1);

        % 找到最佳替換位置（考慮相鄰工地）
        site_positions = find(position == site);
        best_pos = find_best_position(position, site_positions, new_site);

        if ~isempty(best_pos)
            position(best_pos) = new_site;
            site_counts(new_site) = site_counts(new_site) + 1;
            site_counts(site) = site_counts(site) - 1;
            diff = diff - 1;
        else
            break;
        end
    end
end

% 檢查並修復未滿足需求的工地
while any(site_counts ~= demand_trips)
    shortages = demand_trips - site_counts;
    [~, site_to_add] = max(shortages);
    excesses = site_counts - demand_trips;
    [~, site_to_remove] = max(excesses);

    if shortages(site_to_add) <= 0 || excesses(site_to_remove) <= 0
        break;
    end

    % 尋找最佳交換位置
    positions_to_change = find(position == site_to_remove);
    best_pos = positions_to_change(1);

    position(best_pos) = site_to_add;
    site_counts(site_to_add) = site_counts(site_to_add) + 1;
    site_counts(site_to_remove) = site_counts(site_to_remove) - 1;
end

repaired_position = position;
end



%確認是否沒有相鄰
function best_pos = find_best_position(chromosome, positions, new_site)
if isempty(positions)
    best_pos = [];
    return;
end

best_pos = positions(1);
min_conflicts = inf;

for pos = positions
    % 檢查相鄰位置的衝突
    conflicts = 0;
    if pos > 1 && chromosome(pos-1) == new_site
        conflicts = conflicts + 1;
    end
    if pos < length(chromosome) && chromosome(pos+1) == new_site
        conflicts = conflicts + 1;
    end

    if conflicts < min_conflicts
        min_conflicts = conflicts;
        best_pos = pos;
    end
end
end