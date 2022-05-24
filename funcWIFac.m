function [WIFac] = funcWIFac(x1, y1, x2, y2) 
% %     n_Exp_x = length(x1);
% %     n_Sim_x = length(x2);
%     
% %     y1 = round(y1, 6); 
% %     y2 = round(y2, 6);
% 
%     often = max(power(y1,2),power(y2,2));
%     exp2nd = (1-max(0, y1
%     
%     f_nsquare_Exp = power(y1,2);   
%     g_nsquare_Sim = power(y2,2);   
%     
%     fg_product = y1.*y2;
%     
%     for i = 1:n_Sim_x
%         if f_nsquare_Exp(i) == 0
%         else if g_nsquare_Sim(i) == 0
%                 numerator_Exp_Sim(i) = 0;
%             else
%                 numerator_Exp_Sim(i) = max(f_nsquare_Exp(i),g_nsquare_Sim(i)).*((1-(max(0,fg_product(i))/max(f_nsquare_Exp(i),g_nsquare_Sim(i)))).^2);   % 분자항 계산
%                 denominator_Exp_Sim(i) = max(f_nsquare_Exp(i),g_nsquare_Sim(i));   % 분모항 계산
%             end
%         end
%     end
%     WIFac = 1-nthroot((sum(numerator_Exp_Sim(:,i))/sum(denominator_Exp_Sim(:,i))),2); 
% %     WIFac = 0;

often = max(power(y2,2),power(y1,2));
[row, ~] = find(often);  % 0이 아닌 index
often = often(row);

exp2nd_num = max(0, y2.*y1);
exp2nd_num = exp2nd_num(row);

exp2nd_denom = often;

exp2nd = (1 - exp2nd_num./exp2nd_denom).^2;

numerator = sum(often.*exp2nd);
denominator = sum(often);
WIFac = 1 - sqrt(numerator/denominator);
end