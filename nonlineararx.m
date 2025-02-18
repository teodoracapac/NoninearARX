%% Load Data
load("iddata-03.mat");

uid = id.InputData;
yid = id.OutputData;
uval = val.InputData;
yval = val.OutputData;
i=0;
nk = 1;       % delay for inputs
m_max = 10;    % max polynomial degree
na_nb_max = 5; % max value for na and nb

best_MSE_id_pred = Inf;
best_MSE_id_sim = Inf;
best_MSE_val_pred = Inf;
best_MSE_val_sim = Inf; %initialized with inf to have a very high beggining of comparison

best_params_id_pred = [0, 0];  
best_params_id_sim = [0, 0];  
best_params_val_pred = [0, 0]; 
best_params_val_sim = [0, 0]; % format in which these will be saved is [na m]; na=nb

% iterate over na = nb and all degrees up to the maximum
for na = 2:na_nb_max
    for m = 3:m_max
        nb = na;  
        dmax = max(na, nb + nk);  % max delay
        
        % generation of all power combinations in a matrix form
        powers = matrixofpowers(na+nb);

        phi_id = [];
        y_id_total = [];

        % build the regressor matrix phi for identification
        for k = dmax:length(yid)
            d = [];
        
            % delayed outputs
            for i = 1:na
                if (k-i > 0) %condition ensures we don't have negative indices
                    d = [d, yid(k-i)];
                end
            end
        
            % delayed inputs
            for j = 0:(nb-1)
                if (k-nk-j > 0)
                    d = [d, uid(k-nk-j)];
                end
            end

            cross_terms = []; %computed cross products will be stored here
            for row = 1:size(powers, 1) % iterate over all rows of the powers matrix
                term = 1; 
                for index_d = 1:length(d) %compute the terms for each combination of powers
                    term = term * d(index_d)^powers(row, index_d); %powers=matrix returned by the function
                end
                cross_terms = [cross_terms, term]; % concatenate the cross products to the current row
            end

            row = [d, cross_terms];  %concatenate d and cross products
            phi_id = [phi_id; row]; 
            y_id_total = [y_id_total; yid(k)];
        end

        %% Model Identification
        theta = phi_id \ y_id_total;
        y_pred_id = phi_id * theta;

        MSE_id = mean((y_id_total - y_pred_id).^2);
        
        if MSE_id < best_MSE_id_pred
            best_MSE_id_pred = MSE_id;
            best_params_id_pred = [na, m];
        end

        %% Simulation for identification
        y_sim_id = zeros(length(yid), 1);
        y_sim_id(1:dmax) = yid(1:dmax); % initialize first dmax steps with actual values of the output id data

        for k = dmax+1:length(yid)
            d_sim = [];
        
            %delayed outputs
            for i = 1:na
                if (k-i > 0)
                    d_sim = [d_sim, y_sim_id(k-i)];
                end
            end
        
            %delayed inputs
            for j = 0:(nb-1)
                if (k-nk-j > 0)
                    d_sim = [d_sim, uid(k-nk-j)];
                end
            end

            cross_terms_sim = [];
            for row = 1:size(powers, 1)
                term = 1;
                for index_d = 1:length(d_sim)
                    term = term * d_sim(index_d)^powers(row, index_d);
                end
                cross_terms_sim = [cross_terms_sim, term];
            end

            row_sim = [d_sim, cross_terms_sim];
            y_sim_id(k) = row_sim * theta;
        end

        %matching the dimensions
        y_sim_id_final = y_sim_id(dmax:end);

        MSE_id_sim = mean((y_id_total - y_sim_id_final).^2);
        if MSE_id_sim < best_MSE_id_sim
            best_MSE_id_sim = MSE_id_sim;
            best_params_id_sim = [na, m];
        end

        %% Simulation for validation
        y_val_simulated = zeros(length(yval), 1);
        y_val_simulated(1:dmax) = yval(1:dmax); 

        for k = dmax+1:length(yval)
            d_sim = [];
        
            %delayed outputs
            for i = 1:na
                if (k-i>0)
                    d_sim = [d_sim, y_val_simulated(k-i)];
                end
            end
        
            %delayed inputs
            for j = 0:(nb-1)
                if (k-nk-j>0)
                    d_sim = [d_sim, uval(k-nk-j)];
                end
            end

            cross_terms_sim_val = [];
            for row = 1:size(powers, 1)
                term = 1;
                for index_d = 1:length(d_sim)
                    term = term * d_sim(index_d)^powers(row, index_d);
                end
                cross_terms_sim_val = [cross_terms_sim_val, term];
            end

            row_sim_val = [d_sim, cross_terms_sim_val];
            y_val_simulated(k) = row_sim_val * theta;
        end

        MSE_val_simulation = mean((yval - y_val_simulated).^2);
        
        if MSE_val_simulation < best_MSE_val_sim
            best_MSE_val_sim = MSE_val_simulation;
            best_params_val_sim = [na, m];
        end

        %% Validation Prediction
        y_pred_val = zeros(length(yval), 1);

        for k = dmax:length(yval)
            d_pred = [];
        
            %delayed outputs
            for i = 1:na
                if (k-i > 0)
                     d_pred = [d_pred, yval(k - i)];
                end
            end

            %delayed inputs
            for j = 0:(nb-1)
                if (k-nk-j > 0)
                    d_pred = [d_pred, uval(k - nk - j)];
                end
            end

            cross_terms_pred_val = [];
            for row = 1:size(powers, 1) % goes across each line(row)
                term = 1;
                for index_d = 1:length(d_pred)
                    term = term * d_pred(index_d)^powers(row, index_d);
                end
                cross_terms_pred_val = [cross_terms_pred_val, term];
            end

            row_pred_val = [d_pred, cross_terms_pred_val];
            y_pred_val(k) = row_pred_val * theta;
        end

        MSE_val_prediction = mean((yval(dmax:end) - y_pred_val(dmax:end)).^2);
        if MSE_val_prediction < best_MSE_val_pred
            best_MSE_val_pred = MSE_val_prediction;
            best_params_val_pred = [na, m];
        end
        
    if (na==4 && m==1) % best model for prediction identification according to the min MSE
        subplot(221)
        plot(y_id_total, 'b'); hold on;
        plot(y_pred_id, 'r');
        title('Identification Data: Actual vs Predicted');
        xlabel('Time Steps');
        ylabel('Output');
        legend('Actual Output', 'Predicted Output');
        grid on;
    end
    if (na==3 && m==1) % best model for simulated identification according to the min MSE
        subplot(222)
        plot(y_id_total, 'b'); hold on;
        plot(y_sim_id_final, 'm');
        title('Identification Data: Actual vs Simulated');
        xlabel('Time Steps');
        ylabel('Output');
        legend('Actual Output', 'Simulated Output');
        grid on;
    end
    if (na==2 && m==1) % best model for prediction validation according to the min MSE
        subplot(223)
        plot(yval, 'b'); hold on;
        plot(y_pred_val, 'g');
        title('Validation Data: Actual vs Predicted');
        xlabel('Time Steps');
        ylabel('Output');
        legend('Actual Output', 'Predicted Output');
        grid on;
    end
    if (na==1 && m==1) % best model for simulated validation according to the min MSE
        subplot(224)
        plot(yval, 'b'); hold on;
        plot(y_val_simulated, 'r');
        title('Validation Data: Actual vs Simulated');
        xlabel('Time Steps');
        ylabel('Output');
        legend('Actual Output', 'Simulated Output');
        grid on;
    end
    end   
end

%% Display Minimum MSE Results
disp(['Minimum MSE (Identification - Prediction): ', num2str(best_MSE_id_pred), ...
    ' achieved at na = ', num2str(best_params_id_pred(1)), ', degree = ', num2str(best_params_id_pred(2))]);

disp(['Minimum MSE (Identification - Simulation): ', num2str(best_MSE_id_sim), ...
    ' achieved at na = ', num2str(best_params_id_sim(1)), ', degree = ', num2str(best_params_id_sim(2))]);

disp(['Minimum MSE (Validation - Prediction): ', num2str(best_MSE_val_pred), ...
    ' achieved at na = ', num2str(best_params_val_pred(1)), ', degree = ', num2str(best_params_val_pred(2))]);

disp(['Minimum MSE (Validation - Simulation): ', num2str(best_MSE_val_sim), ...
    ' achieved at na = ', num2str(best_params_val_sim(1)), ', degree = ', num2str(best_params_val_sim(2))]);

%% Function: Generate Cross Product Powers
function [pow] = matrixofpowers(m)
    % Generate all possible combinations of powers for the delay vector
    pow = zeros(2^m, m);
    for i = 1:2^m
        binaryRow = dec2bin(i-1, m) - '0'; 
        pow(i, :) = binaryRow;
    end
end
