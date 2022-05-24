%%%% User input %%%%
Results_tag = date;  
%%%%%%%%%%%%%%%%%%%%

folderName = strcat("Results_", Results_tag);
mkdir(folderName);
save(strcat(folderName, "/Results", ".mat"), 'Result_X1', 'Result_X2', 'Result_X3', 'Result_X4', 'Result_X5', 'Result_Y');
savefig(strcat(folderName, "/Resulting_figure", ".fig"));
save(strcat(folderName, "/Resulting_Workspace", ".mat"));