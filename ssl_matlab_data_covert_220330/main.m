%% Embry dataset
data_path = [pwd, '../../../data/Embry/InclineExperiment.mat'];
load(data_path)

file = "export.h5";
delete export.h5

h5create(file, '/fields/subject_details', [6 1], Datatype='string')
fields = sub_data.subjectdetails(1:end, 1);
h5write(file, '/fields/subject_details', fields)

h5create(file, '/fields/subject_details_units', [6 1], Datatype='string')
fields = sub_data.subjectdetails(1:end, 3);
h5write(file, '/fields/subject_details_units', fields)

h5create(file, '/fields/subject_details_units', [6 1], Datatype='string')
fields = sub_data.subjectdetails(1:end, 3);
h5write(file, '/fields/subject_details_units', fields)


subs = fieldnames(Continuous);
for i_sub = 1:numel(subs)
    sub = subs{i_sub};
    sub_data = Continuous.(sub);
    h5create(file, ['/' sub '/subject_details'], [6 1])
    values = cell2mat(sub_data.subjectdetails(1:end, 2));
    h5write(file, ['/' sub '/subject_details'], values)

    trials = fieldnames(Continuous.(sub));
    for i_trial = 2:numel(trials)
        trial = trials{i_trial};
        trial_data = sub_data.(trial);
        data_array = transform_data_data(trial_data.kinematics.markers);

%         h5create(file, ['/' sub trial '/marker'], )
%         values = cell2mat(sub_data.subjectdetails(1:end, 2));
%         h5write(file, ['/' sub '/subject_details'], values)

    end
end



% function data_array = transform_data_data(marker_struct)
%     marker_col_num = 2*8*3;
%     imu_col_num = ;
%     data_array = zeros([6000, ]);
%     i_col = 1;
%     for side = {'left', 'right'}
%         side_data = marker_struct.(cell2mat(side));
%         for marker = reshape(fieldnames(side_data), [1, 8])
%             data_array(1:end, i_col:i_col+2) = side_data.(cell2mat(marker));
%             i_col = i_col + 3;
%         end
%     end
% end








