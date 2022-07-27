classdef DataSet
    properties
        emg_fields
        emg_data
        anthro_fields
        anthro_data
        marker_fields
        marker_data
        
    end
   
    methods
        function initialization

        end

        function obj = classA(a,b,c,d)
            if nargin == 0         
            end
            obj.a = a;
            obj.b = b;
            obj.c = c;
            obj.d = d;
        end
    end

end