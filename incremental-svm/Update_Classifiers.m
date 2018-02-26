function new_classifiers = Update_Classifiers(classifiers)


for i = 1:length(classifiers)
    
    new_classifiers(i) = svmincrement(classifiers(i));
    
end