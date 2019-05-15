library(progress)
library(gdata)
#need to add this line in console
#source('C:/Users/Marco/PycharmProjects/DFKI/hiwi/deepEx/DeepEX-master/Fourtheta.R')
cif = read.csv('C:/Users/Marco/PycharmProjects/DFKI/hiwi/deepEx/DeepEX-master/M3C_other.csv') #-----change
y=cif
ignore_col = 6 #3 for cif, 6 for m3 -----change
frequency=4 #try with 12,7,4 -----change
rows= dim(y)[1]
col = dim(y)[2]
pb <- progress_bar$new(total = rows)
per = 0.25
stat_length = as.integer((col-ignore_col)*per) 
final_comb = matrix(,nrow=1,ncol=(col-ignore_col-stat_length))
for (i in 1:rows) {
  
  
  data = y[i,ignore_col:col]
  tb=table(is.na(data))
  length = tb[["FALSE"]]
  data = data[,1:length]
  d_l = dim(data)[2]
  s_l =as.integer(d_l*per)
  
  data_fit  = data[,s_l:(2*s_l)] #-------change
  data_nn = data[,(2*s_l+1):d_l] #-------change
  
  horizon = 3 #-----change
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
write.csv(final_comb,"C:/Users/Marco/PycharmProjects/DFKI/hiwi/deepEx/DeepEX-master/theta_25_h3_m3o4.csv") #need to add path -----change
