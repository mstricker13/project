library(progress)
library(gdata)
#need to add this line in console
#source('C:/Users/Marco/PycharmProjects/DFKI/hiwi/deepEx/DeepEX-master/Fourtheta.R')
cif = read.csv('C:/Users/Marco/PycharmProjects/DFKI/project/hiwi/deepEx/DeepEX-master/nn3_no_meta_test_mirror.csv', header = FALSE) #-----change
y=cif
#number of columns which contain meta information and should be ignored
ignore_col = 3 # 0 for old nn3 #3 for cif, 6 for m3 -----change
ignore_col_index = ignore_col + 1 #see ToDo 2
frequency= 12 #try with 12,7,4 -----change
#rows and columns of input
rows= dim(y)[1]
col = dim(y)[2]
pb <- progress_bar$new(total = rows)
per = 0.25
#number of values which should be removed to train theta, i.e. remove meta data columns and percentage many columns
stat_length = as.integer((col-ignore_col)*per)
#final number of values after removing the meta and train data
final_comb = matrix(,nrow=1,ncol=(col-ignore_col-stat_length))
for (i in 1:rows) {
  
  #ignore empty fields
  data = y[i,ignore_col_index:col]
  tb=table(is.na(data))
  length = tb[["FALSE"]]
  data = data[,1:length]
  d_l = dim(data)[2]
  s_l =as.integer(d_l*per)
  
  data_fit  = data[,1:s_l] #-------change
  data_nn = data[,(s_l+1):d_l] #-------change
  
  horizon = 18 #-----change
  out = FourTheta(ts(t(data_fit),frequency=frequency),horizon)
  temp = t(as.numeric(out$mean))
  len = length(data_nn)
  #mat = matrix(,nrow=1,ncol=len)
  
  loop = len%/%horizon
  rem = len%%horizon
  for (z in 1:(loop)){
    data_fit = data[,1:s_l+(z*horizon)]
    out=FourTheta(ts(t(data_fit),frequency=frequency),horizon)
    temp = cbind(temp,t(as.numeric(out$mean)))
    
  }
  final = t(temp[1:len])
  
  # final = cbind(t(fitted),t(forecast))
  #final=t(fitted)
  temp_1 = length(final)
  tmp1 = col-ignore_col-temp_1-stat_length
  if (tmp1 != 0){
    mat = matrix(,nrow=1,ncol=tmp1)
    final1 = cbind(final,mat)
  } else {
    final1 = final
  }
  #mat = matrix(,nrow=1,ncol=tmp1)
  # final1 = cbind(final,mat)
  final_comb = rbind(final_comb,final1)
  #print(i)
  pb$tick()
}
write.csv(final_comb,"C:/Users/Marco/PycharmProjects/DFKI/project/hiwi/deepEx/DeepEX-master/theta_25_hT_nn3_12_test.csv") #need to add path -----change
