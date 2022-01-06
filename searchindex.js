Search.setIndex({docnames:["chapters/api/ccp","chapters/api/database","chapters/api/index","chapters/api/rf","chapters/api/test","chapters/api/utils","chapters/api/waveform","chapters/ccp/ccpstackobject","chapters/ccp/index","chapters/ccp/intro","chapters/creation/index","chapters/creation/requestclass","chapters/introduction/installation","chapters/introduction/introduction","chapters/plotting/ccp","chapters/plotting/index","chapters/plotting/receiverfunctions","chapters/receiver_functions/hdf5","chapters/receiver_functions/index","chapters/receiver_functions/methods","chapters/receiver_functions/sac","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["chapters/api/ccp.rst","chapters/api/database.rst","chapters/api/index.rst","chapters/api/rf.rst","chapters/api/test.rst","chapters/api/utils.rst","chapters/api/waveform.rst","chapters/ccp/ccpstackobject.rst","chapters/ccp/index.rst","chapters/ccp/intro.rst","chapters/creation/index.rst","chapters/creation/requestclass.rst","chapters/introduction/installation.rst","chapters/introduction/introduction.rst","chapters/plotting/ccp.rst","chapters/plotting/index.rst","chapters/plotting/receiverfunctions.rst","chapters/receiver_functions/hdf5.rst","chapters/receiver_functions/index.rst","chapters/receiver_functions/methods.rst","chapters/receiver_functions/sac.rst","index.rst"],objects:{"pyglimer.ccp":{ccp:[0,0,0,"-"],io:[0,0,0,"-"]},"pyglimer.ccp.ccp":{CCPStack:[0,1,1,""],PhasePick:[0,1,1,""],init_ccp:[0,3,1,""],read_ccp:[0,3,1,""]},"pyglimer.ccp.ccp.CCPStack":{compute_kdtree_volume:[0,2,1,""],compute_stack:[0,2,1,""],conclude_ccp:[0,2,1,""],create_vtk_mesh:[0,2,1,""],explore:[0,2,1,""],map_plot:[0,2,1,""],multicore_stack:[0,2,1,""],pick_phase:[0,2,1,""],plot_bins:[0,2,1,""],plot_cross_section:[0,2,1,""],plot_volume_sections:[0,2,1,""],query_bin_tree:[0,2,1,""],write:[0,2,1,""]},"pyglimer.ccp.ccp.PhasePick":{plot:[0,2,1,""]},"pyglimer.ccp.compute":{bin:[0,0,0,"-"]},"pyglimer.ccp.compute.bin":{BinGrid:[0,1,1,""],fibonacci_sphere:[0,3,1,""]},"pyglimer.ccp.compute.bin.BinGrid":{compute_bins:[0,2,1,""],query_bin_tree:[0,2,1,""],query_station_tree:[0,2,1,""],station_tree:[0,2,1,""]},"pyglimer.ccp.io":{load_rawrf:[0,3,1,""],load_tracefile:[0,3,1,""],load_velocity_model:[0,3,1,""]},"pyglimer.ccp.plot_utils":{line_buffer:[0,0,0,"-"],plot_cross_section:[0,0,0,"-"],plot_line_buffer:[0,0,0,"-"]},"pyglimer.ccp.plot_utils.line_buffer":{line_buffer:[0,3,1,""]},"pyglimer.ccp.plot_utils.plot_cross_section":{get_ax_coor:[0,3,1,""],plot_cross_section:[0,3,1,""]},"pyglimer.ccp.plot_utils.plot_line_buffer":{plot_line_buffer:[0,3,1,""]},"pyglimer.database":{asdf:[1,0,0,"-"],rfh5:[1,0,0,"-"],stations:[1,0,0,"-"]},"pyglimer.database.asdf":{rewrite_to_hdf5:[1,3,1,""],write_st:[1,3,1,""],writeraw:[1,3,1,""]},"pyglimer.database.rfh5":{DBHandler:[1,1,1,""],RFDataBase:[1,1,1,""],all_traces_recursive:[1,3,1,""],convert_header_to_hdf5:[1,3,1,""],read_hdf5_header:[1,3,1,""]},"pyglimer.database.rfh5.DBHandler":{add_rf:[1,2,1,""],get_coords:[1,2,1,""],get_data:[1,2,1,""],walk:[1,2,1,""]},"pyglimer.database.stations":{redownload_missing_statxmls:[1,3,1,""]},"pyglimer.rf":{create:[3,0,0,"-"],deconvolve:[3,0,0,"-"],moveout:[3,0,0,"-"]},"pyglimer.rf.create":{RFStream:[3,1,1,""],RFTrace:[3,1,1,""],createRF:[3,3,1,""],obj2stats:[3,3,1,""],read_by_station:[3,3,1,""],read_rf:[3,3,1,""],rfstats:[3,3,1,""]},"pyglimer.rf.create.RFStream":{bootstrap:[3,2,1,""],dirty_ccp_stack:[3,2,1,""],method:[3,2,1,""],moveout:[3,2,1,""],plot:[3,2,1,""],plot_distribution:[3,2,1,""],ppoint:[3,2,1,""],slice2:[3,2,1,""],station_stack:[3,2,1,""],trim2:[3,2,1,""],type:[3,2,1,""],write:[3,2,1,""]},"pyglimer.rf.create.RFTrace":{moveout:[3,2,1,""],plot:[3,2,1,""],ppoint:[3,2,1,""],write:[3,2,1,""]},"pyglimer.rf.deconvolve":{damped:[3,3,1,""],it:[3,3,1,""],multitaper:[3,3,1,""],spectraldivision:[3,3,1,""]},"pyglimer.rf.moveout":{SimpleModel:[3,1,1,""],dt_table:[3,3,1,""],dt_table_3D:[3,3,1,""],earth_flattening:[3,3,1,""],load_model:[3,3,1,""],moveout:[3,3,1,""],ppoint:[3,3,1,""]},"pyglimer.test":{rate:[4,0,0,"-"],synthetic:[4,0,0,"-"],tests:[4,0,0,"-"]},"pyglimer.test.rate":{automatic_rate:[4,3,1,""],rate:[4,3,1,""],sort_rated:[4,3,1,""]},"pyglimer.test.synthetic":{create_R:[4,3,1,""],synthetic:[4,3,1,""]},"pyglimer.test.tests":{decon_test:[4,3,1,""],moveout_test:[4,3,1,""],read_geom:[4,3,1,""],read_raysum:[4,3,1,""],read_rfs_mat:[4,3,1,""],rf_test:[4,3,1,""],test_SNR:[4,3,1,""]},"pyglimer.utils":{create_geom:[5,0,0,"-"],createvmodel:[5,0,0,"-"],geo_utils:[5,0,0,"-"],nextpowof2:[5,0,0,"-"],roundhalf:[5,0,0,"-"],signalproc:[5,0,0,"-"],utils:[5,0,0,"-"]},"pyglimer.utils.create_geom":{create_geom:[5,3,1,""]},"pyglimer.utils.createvmodel":{load_avvmodel:[5,3,1,""],load_gyps:[5,3,1,""]},"pyglimer.utils.geo_utils":{cart2geo:[5,3,1,""],epi2euc:[5,3,1,""],euc2epi:[5,3,1,""],gctrack:[5,3,1,""],geo2cart:[5,3,1,""],geodiff:[5,3,1,""],geodist:[5,3,1,""],reckon:[5,3,1,""]},"pyglimer.utils.nextpowof2":{nextPowerOf2:[5,3,1,""]},"pyglimer.utils.roundhalf":{roundhalf:[5,3,1,""]},"pyglimer.utils.signalproc":{convf:[5,3,1,""],corrf:[5,3,1,""],filter:[5,3,1,""],gaussian:[5,3,1,""],noise:[5,3,1,""],resample_or_decimate:[5,3,1,""],ricker:[5,3,1,""],sshift:[5,3,1,""]},"pyglimer.utils.utils":{chunks:[5,3,1,""],create_bulk_str:[5,3,1,""],download_full_inventory:[5,3,1,""],dt_string:[5,3,1,""],get_multiple_fdsn_clients:[5,3,1,""],save_raw:[5,3,1,""],save_raw_mseed:[5,3,1,""]},"pyglimer.waveform":{download:[6,0,0,"-"],errorhandler:[6,0,0,"-"],preprocess:[6,0,0,"-"],preprocessh5:[6,0,0,"-"],qc:[6,0,0,"-"],request:[6,0,0,"-"],rotate:[6,0,0,"-"]},"pyglimer.waveform.download":{download_small_db:[6,3,1,""],downloadwav:[6,3,1,""],get_mseed_storage:[6,3,1,""],wav_in_asdf:[6,3,1,""],wav_in_db:[6,3,1,""]},"pyglimer.waveform.errorhandler":{NoMatchingResponseHandler:[6,3,1,""],redownload:[6,3,1,""],redownload_statxml:[6,3,1,""]},"pyglimer.waveform.preprocess":{SNRError:[6,4,1,""],StreamLengthError:[6,4,1,""],preprocess:[6,3,1,""],write_info:[6,3,1,""]},"pyglimer.waveform.preprocessh5":{SNRError:[6,4,1,""],StreamLengthError:[6,4,1,""],compute_toa:[6,3,1,""],preprocessh5:[6,3,1,""]},"pyglimer.waveform.qc":{qcp:[6,3,1,""],qcs:[6,3,1,""]},"pyglimer.waveform.request":{Request:[6,1,1,""]},"pyglimer.waveform.request.Request":{download_eventcat:[6,2,1,""],download_waveforms:[6,2,1,""],download_waveforms_small_db:[6,2,1,""],preprocess:[6,2,1,""]},"pyglimer.waveform.rotate":{rotate_LQT_min:[6,3,1,""],rotate_PSV:[6,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:exception"},terms:{"01st":5,"02nd":5,"100":1,"1000":3,"1024":1,"10hz":1,"10km":3,"10th":[0,6],"11th":1,"12th":[1,3],"14056":5,"14th":5,"15th":6,"16th":3,"180":3,"18th":[5,6],"1977":6,"1999":[3,6],"19th":[4,6],"1st":[3,4],"200":[0,3,14],"2006":6,"2009":[3,11],"2011":11,"2013":3,"2016":3,"2019":[0,3,4,5,6],"2020":[0,1,3,4,5,6],"2021":[0,1,3,4,5,6],"2022":[1,6],"20444":5,"20472":5,"20th":5,"211":14,"21st":[0,1,3,4,5,6],"21th":6,"22nd":6,"23rd":4,"250":1,"25th":[1,3,4,5,6],"26th":5,"27th":6,"289":17,"28th":6,"2nd":0,"300":[0,3],"30deg":7,"360":3,"36056":5,"400":3,"521":1,"57735026918962573":0,"5th":[1,6],"750":3,"753":16,"7th":3,"800":[3,5,16],"86602540378443871":0,"boolean":1,"byte":1,"case":[3,6,11],"class":[0,1,3,6,7,10,17,18,21],"default":[0,1,3,4,5,6],"export":0,"final":[0,11],"float":[0,1,3,4,5,6],"function":[0,1,3,4,5,6,7,8,9,11,13,15,18,19],"import":[0,5,11,14,16,17],"int":[0,3,4,5,6],"long":[0,4,5],"new":[0,1,3,5,6,11],"public":[0,1,3,4,5,6,11,17,19],"return":[0,1,3,4,5,6,17],"static":0,"true":[0,1,3,4,5,6,7,14,16],"while":[0,1,11,18],AND:3,Axes:[0,3],BUT:3,FOR:[0,3],For:[0,1,3,4,5,6,7,11,19],Has:[3,5],Its:0,NOT:3,Not:[0,5],One:[0,4,6],PPS:3,PWS:3,RFs:[0,3,4,6,16,19],SKS:[6,11],ScS:[6,11],THE:3,That:0,The:[0,1,3,4,5,6,7,10,12,14,17,19,21],Then:[0,4,7,11],There:[0,3],These:[5,18],USE:[0,3],Use:[0,3,6,11],Used:[4,6],Useful:5,Uses:0,Using:[0,7],WITH:3,Will:[0,1,3,4,5,6],With:6,_ax:[0,3],_close:17,_db:6,_hl:1,_load:0,about:[0,3,5,7],abov:[1,3,5,17],accept:[1,3,5,6,17],access:[1,5,11,17],accord:1,account:0,accur:[0,3,5],acronym:6,action:3,activ:12,actual:[1,3,7,14,15],adapt:[1,6],add:[1,3,17],add_rf:[1,17],added:[1,3,7],adding:3,addit:[1,3,4],addition:11,advantag:[6,11],advis:3,affect:0,aforement:[5,7],after:[3,4,5,6,7,19],afterward:[1,5,17],against:[3,19],aggreg:1,aim:13,algorithm:[1,3,5],alias:5,all:[0,1,3,4,5,6,7,11,15,16,17,18,19],all_traces_recurs:1,allow:[0,1,5,6,7,11],along:[5,6],alpha:4,alreadi:[1,3,4,6,11],also:[0,3,4,5,6,7,11,16],alter:1,altern:[0,3,5,11],alwai:[0,1,11],ammon:[3,6],amount:[0,5,7,13],amplitud:[0,3,4,5],andrew:4,angl:[0,3,6],angular:[0,3],ani:[0,3,11,17],anoth:[6,11],anymor:0,anywai:3,apart:5,api:21,appar:6,appear:3,append:[0,1,3,19],append_pp:0,appli:20,applic:1,approach:[0,3],approxim:[1,3],approximaton:0,apr:4,april:[0,3,4,6],arang:14,archiv:[0,7],area:[0,3,7],aren:0,arg:0,argument:[3,7,16,17],aris:3,around:[0,6,7],arrai:[0,1,3,4,5,6,13,14],arraylik:[0,3,4,5],arriv:[3,6],artist:0,artsi:16,asdf:[2,5,6],associ:[1,3,5,17,21],assort:4,assum:3,assumpt:3,attemp:1,attempt:3,attribdict:3,attribut:[0,1,3,6],august:[1,4,5],author:[0,1,3,4,5,6],autom:[3,13],automat:[4,5,7,17],automatic_r:4,avail:[1,3,5,7,11,16,18,21],aval:16,averag:[0,5,6,7,9],averagevelmodel:5,avoid:[3,5,17],avp:6,avs:6,awai:0,awar:3,ax1:14,ax2:14,axes:[0,3,16],axi:3,azimuth:[0,3,5,6,19],back:[0,3,6],back_azimuth:3,backazimuth:4,backend:[0,6],backzimuth:5,bandwidth:3,base:[0,1,3,6,7,11,13,18],baz:[0,3,4],bazv:5,bbox:0,bear:5,been:[0,1,4,6],befor:[0,1,3,5,6,11],begin:[3,6],behaviour:17,being:[0,6],below:[0,1,3,11,17,19],benchmark:3,best:0,better:[6,11],between:[0,1,3,4,5,6,7,16],big:[0,5],bin:[2,3,8],binari:0,bingrid:[0,7],binrad:0,bit:3,black:3,block:1,blur:7,bool:[0,1,3,4,5,6],bootstrap:3,both:[6,11,18,21],bound:[0,1,3],boundari:[0,14],box:0,broadest:6,buffer:0,build:14,built:[14,18],cach:1,calcul:[3,5,6],call:[0,7,17],can:[0,1,3,4,5,6,7,9,11,13,14,15,16,17,18,19,20],cannot:[4,7],cap:0,capabl:21,carlo:3,cart2geo:5,cartesian:[0,3,5],cartesioan:3,cartopi:0,cast:6,catalog:[1,6,11],catalogu:[1,6,11],catfil:1,caus:[0,17],cbaz:0,ccp:[2,3,4,9,13,15,19,21],ccp_iu_hrv:14,ccpstack:[0,8,14,21],ccpstream:0,celev:0,cell:0,center:[0,3],centr:0,certain:[1,4,5,6,9,10],chang:3,channel:[3,5,6,17],charg:3,check:[1,3,4,6],child:1,choic:[3,6,11],choos:[1,11],chord:5,chosen:[0,3,11],chunk:[0,1,5],circl:5,circular:[0,6],claim:3,clean:[3,16],clear:3,client:[1,5,6,11],clip:6,clone:12,close:[0,7,17],closest:[0,5],cmap:0,coars:7,code:[0,1,3,4,5,6,7,12,15,17],coher:3,collect:16,collis:1,colormap:0,colour:0,column:[3,6],com:12,combin:3,come:[5,11,17],command:0,common:[0,8,17,21],commonli:7,compar:3,compat:[0,6,7],compil:[0,5,6],complet:5,complex:3,complexmodel:5,compon:[3,4,5,6,11],compress:1,comput:[2,3,5,6,7,8,11,14,17,19,21],computation:0,compute_bin:0,compute_kdtree_volum:0,compute_stack:[0,6,7],compute_toa:6,con:[3,4],concern:[7,11],conclude_ccp:[0,7],conda:12,condit:3,conduct:5,config:0,connect:[0,3],consecut:3,consid:[0,6],consist:11,consoel:0,consol:0,constant:[3,4,5],constant_dist:5,constantdist:5,constantli:4,constrain:6,construct:4,consult:19,conta:0,contain:[0,1,3,4,5,6,7,11],content:0,context:[1,17],continent:11,continu:6,continue_download:6,contract:3,contrast:6,control:[3,4,6,7],conveni:[0,3,11],convent:1,converg:3,convers:[0,3,8,21],convert:[1,3,5,6,13],convert_header_to_hdf5:1,convf:5,convolut:5,convolv:[3,4,5],coord:0,coordin:[0,1,3,4,5,6,11,17],copi:3,copyleft:[0,1,3,4,5,6],copyright:[0,1,3,4,5,6],core:[0,1,3,5,6],corner:0,correct:[3,5],correl:[1,5],correspond:[0,1,3,4,5,6],corrf:5,corrstream:[1,17],corrtrac:1,corrupt:17,cos:7,cosd:0,could:[1,3,11,17],count:[0,1,17],cours:11,cover:[0,7],coverag:0,cpp:0,crayp:0,creat:[0,1,2,4,5,6,8,9,11,12,13,14,16,17,18,19],create_bulk_str:5,create_geom:2,create_r:4,create_vtk_mesh:0,createrf:3,createvmodel:2,creation:[1,5,6,11,18,21],crf:0,crit:[4,6],criteria:[4,6],criterion:6,critic:[6,11],cross:[0,5,15],crosscorr:5,crust:5,cslat:0,cslon:0,cst:17,cumul:0,current:[3,5],curv:5,cut:[6,16],cylindr:7,damag:3,damp:[3,4,6],dampedf:[3,4,6],dat:[0,3,19],data:[0,1,3,4,5,6,7,10,11,13,16,18,19,21],databas:[0,2,3,6,9,10,16,17],datacent:6,dataloss:17,dataset:[0,1,6],date:6,dbh:17,dbhandler:[1,17],deal:3,debug:[3,6,11],dec:1,decemb:6,decid:[0,7,17],decim:[1,5],decis:0,decod:4,decon_meth:4,decon_test:4,deconmeth:[6,11],deconvoltuion:[3,4],deconvolut:[3,4,5,6,11],deconvolv:[2,4],decreas:[3,7],decrib:0,dedic:9,defin:[0,1,3,5,6,11,17],deg2km:0,deg:[1,3,4,5,6],degre:[0,3,5,6],delai:3,delet:6,delta:[0,3],deltabin:0,demean:6,denomin:3,dens:7,deoncolut:3,depend:[0,1,3,4,5,6,7,9,16,19],dependentor:3,deprec:0,depth:[0,3,4,19],depthext:[0,14],depthrang:0,describ:[3,5],descript:[0,6],desir:[1,5,7],detail:[1,7,16],determin:[1,3,4,17],detrend:6,develop:[0,1,3,4,5,6,12],deviat:[3,4,5],diagram:19,dic:3,dict:[1,3,5,6],dictionari:[3,5,6,17],diff:4,differ:[0,1,3,5,6,11,17,19],digit:[1,3,7],dimension:0,dip:4,direct:6,directli:[0,1,17],directori:[0,1,3,4,5,6,11,18],dirti:3,dirty_ccp_stack:[3,19],discard:[0,7,11],disclaim:3,discontinu:4,disk:[1,5,11],displai:0,dist:5,distanc:[0,3,5,6,7,16,19],distance_bin:0,distribut:[0,3,5,19],divid:0,divis:[1,3,4,6,11],dlon:3,dmc:6,document:[0,1,3,6,21],doe:[0,1,11],doing:20,domain:[0,3,4,5,6,11,19],don:[3,6],done:[4,6,7,14,18],doubl:3,dowloand:6,down:3,download:[1,2,5,10,11,13,16,21],download_eventcat:[6,11],download_full_inventori:5,download_small_db:6,download_waveform:[6,11],download_waveforms_smal:6,download_waveforms_small_db:[6,11],downloadwav:6,downsampl:11,downsid:11,downweight:0,dpi:[0,3],driver:1,dt_string:5,dt_tabl:3,dt_table_3d:3,due:[0,1,17],dure:1,dynam:6,dzf:3,each:[0,4,5,6,7],earlier:[4,17],earliest:[1,6,11],earth:3,earth_flatten:3,easili:3,east:4,edist:0,edit:12,edu:[0,5],effect:3,effici:[0,7],either:[0,1,3,4,6,11,19],elaps:5,element:0,elev:[0,1,3],els:[1,5,6,7],empi:0,empti:[0,1,7],enabl:7,end:[0,1,3,6],endtim:[3,5,6,11],endus:5,energi:6,enough:[0,7],entir:0,entri:0,enumer:3,env:12,environ:12,environemt:12,epi2euc:5,epi:[0,5],epicentr:[0,3,5,6,16,19],epilimit:[3,16],equal:[0,4,5,7],equat:3,errand:0,error:[3,6,11],errorhandl:2,essenti:[3,5],estim:3,etc:[3,16],euc2epi:5,euc:5,euclidean:[0,3,5],euclidian:[0,3],eulenfeld:[3,18],evalu:4,even:[5,11,17],evenli:5,event:[1,3,5,6,11,17],event_cat:6,event_catalog:11,event_coord:6,ever:17,everi:[5,17],everyth:[0,11],evt:[5,6],evt_subdir:[6,11],evt_tim:[1,17],evtcat:[6,11],evtfil:6,evtloc:6,exact:15,exactli:5,exampl:[1,11,15,16,17,19],except:[3,4,6],execut:0,exist:[1,6,11],exot:11,expect:0,expens:7,experienc:6,experiment:19,explain:8,explan:[1,5],explor:0,express:3,extens:[0,9],extent:[0,3],extra:1,extract:[3,5],face:7,fact:5,factor:[0,3],fail:1,failsaf:0,fair:0,fairli:[1,12],fall:6,fals:[0,1,3,4,5,6],far:0,fast:5,fast_redownload:6,faster:[5,6,11],fastest:[0,1,7],fdsn:[5,6,10,11],fdsn_client:5,februari:[1,3,4,6],feel:16,few:[0,6,11,14],fewer:6,fft:3,fibonacci:0,fibonacci_spher:0,field:0,figur:[0,3,16],file:[0,1,3,4,5,6,7,11,17,18,19],fileformat:11,filehandl:6,filenam:[0,3,4,5,11,14],filt:0,filter:[0,1,3,4,5,6,11],finalis:7,find:[0,1,3,5,6,7,19],fine:7,finer:0,finish:6,first:[0,3,5,6,10,11,14],fit:[0,1,3],fiull:14,fix:3,flag:0,flat:3,flatten:[3,5],flatter:0,flip:[3,5],flipxi:3,flush:1,fmt:[0,14],fname:3,fnmatch:1,folder:[0,1,4,5,6,11],follow:[0,3,5,6,11,16,17],form:[0,3,5,6],format:[0,1,3,4,5,6,7,11,18,21],found:[0,1,6],four:11,fqd:[3,4,6],framework:13,frederiksen:4,free:[1,3],freq:3,frequenc:[3,4,5,6],fridai:[0,1,3,5,6],friendli:13,from:[0,1,3,4,5,6,7,8,10,11,13,14,16,17,19,21],fs_persist:1,fs_strategi:1,fs_threshold:1,fsm:1,full:[0,1,3,4,5,14],fulli:1,fullload:11,furnish:3,futur:[0,3],gaussian:[3,5],gctrack:5,gener:[0,1,3,4,5,6,18],geo2cart:5,geo:0,geo_util:2,geoax:[0,14],geoaxessubplot:0,geocooord:0,geocoord:[0,7],geodiff:5,geodist:5,geograph:[0,5,9],geoloc:0,geolog:0,geologi:0,geom:[4,5],geom_fil:4,geometri:[4,5],get:[0,1,8,12,18,21],get_ax_coor:0,get_config:1,get_coord:[1,17],get_data:[1,17],get_mseed_storag:6,get_multiple_fdsn_cli:5,get_stations_bulk:5,get_waveforms_bulk:5,gfz:[0,1,3,4,5,6],gist_rainbow:0,git:12,github:12,given:[0,1,3,4,5,6,7,13,16],givenn:0,glimer:[0,4,6,7,13],global:[1,11],gnu:[0,1,3,4,5,6],going:8,gonna:0,good:[6,15],grade:9,gradient:0,grai:0,grant:3,graphic:0,great:5,grid:[0,7],group:1,guid:1,gypsum:[0,3,5,19],gzip3:1,gzipx:1,h5py:[1,17],h_nois:4,had:5,half:[4,5],halflength:5,handi:17,handl:[0,1,11,21],handler:[1,6],has:[0,3,5,6,7,11,17],hash:1,hat:5,havard:6,have:[0,1,3,4,6,7,9,11,14,17,18,19],hc_filt:6,hdf5:[0,1,6,10,18,19,21],head:11,header:[1,3,17],heatmap:0,helffrich:[3,6],help:7,henc:[0,5],here:[0,3,5,7,8,11,13,14,15],herebi:3,hero:21,hierach:[1,11],high:[3,6,7],highcof:0,highcut:6,higher:[0,5,7],highest:1,highpass:6,highpassf:0,hit:[0,5],hoizont:4,hold:[1,5,6,17,18],holder:3,hope:5,horizont:[3,4,5,6],hour:5,how:[0,1,3,5,7,8,11,19],howev:[0,5,7,11],hrv:[1,3,16],htab:3,html:[0,1,3,4,5,6],hto:1,http:[0,1,3,4,5,6,12],iasp91:[0,3,19],idea:3,ideal:1,identifi:[3,5,17],ignor:[0,6],illum:0,illumin:[0,3,7,14],imag:[7,13,16,18],immit:0,implement:[0,5,7],impli:3,importantli:11,improv:3,impuls:[3,4,5],in_format:0,incid:[0,3,6],includ:[0,1,3,5,14],inclus:1,incom:[0,6],incompat:0,incorpor:0,increas:[0,1,7],independ:4,index:[0,3,21],indic:[0,1],individu:3,info:[0,3,6,11],inform:[0,1,3,5,6,11,17],inheret:1,init_ccp:[0,7],initi:3,initialis:[0,6,7,11],input:[0,1,3,4,5,6],insert:11,insid:7,instal:[5,21],instanc:3,instantli:3,instead:[0,1,3,11],instrument:6,integ:[4,5],inter:0,interbin:7,interfer:13,intermezzo:7,intern:0,internet:0,interpol:[0,3,5,14],interrupt:[3,6],interv:[0,3,4,5,6],introduc:7,introduct:21,inv:5,invalid:[5,6],inventori:[1,3,5,6],invert:3,iri:[5,6,11],irrelev:6,isn:0,issu:[6,7],it_max:3,iter:[0,1,3,4,5,6,11,17],its:[1,16],itself:[3,7],januari:[1,6],job:0,joblib:[0,5,6],jump:3,june:3,jupyt:[15,19],just:[0,3,5,7,11,17,18],kanamori:[0,3],kathrin:3,kdtree:[0,14],keep:[0,7],keep_empti:0,keep_empty_trac:3,keep_wat:0,kei:[1,17],keyboard:4,keyword:1,kind:3,know:7,known:6,kwarg:[0,3,11],lab:4,label:[0,14,16],lag:3,langston:6,larg:[0,11,13],larger:0,last:[0,1,3,4,5,6],lat0:14,lat1:[0,5,14],lat2:[0,5],lat:[0,3,5,14],latb:[0,3,5],later:[0,1,17,19],latest:[1,6,11],latitud:[0,1,5,6],latsl:[0,14],latter:[3,11],layer:3,lead:[0,5,7,17,18],learn:11,least:1,left:[0,3,6],legaci:7,legal:1,len:3,length:[3,4,5,6],less:[0,6],lesser:[0,1,3,4,5,6],letter:4,level:[1,3,4,6,11],liabil:3,liabl:3,librari:[1,21],libver:1,licens:[0,1,3,4,5,6],ligorria:[3,6],like:[0,1,3,5,7,11,14,17],lim:3,limit:[0,3,6,7,16],line:[0,3],line_buff:2,linear:[0,3],linearli:0,linewidth:[3,16],linspac:[5,14],list:[0,1,3,4,5,6,11,17,19],litho1:[5,6],lithospher:6,load:[0,3,11,17],load_avvmodel:5,load_gyp:5,load_model:3,load_rawrf:0,load_tracefil:0,load_velocity_model:0,loader:11,locat:[0,1,3,5,6,17],log:[0,6,11],log_fh:6,log_subdir:[6,11],logdir:[0,6],logger:6,logic:17,loglvl:[6,11],lon0:14,lon1:[0,5,14],lon2:[0,5],lon:[0,3,5,14],lonb:[0,3,5],longer:[0,3,7],longest:0,longitud:[0,1,3,5,6],longitudin:3,lonsl:[0,14],look:[0,1,3,5,15],loop:5,lost:0,lot:[0,5],low:[5,6,7],lowco:6,lowcof:0,lower:[0,1,3,6,7],lowest:11,lowpassf:0,lqt:[6,11],lrf:3,lru:1,lsawad:[0,5],lst:5,luca:[0,5],magnitud:[6,11],mai:[3,5,6],main:3,mainli:5,make:[0,3,4,5,6],maku:[0,1,3,4,5,6,13],manag:[1,3,11,17],manipul:17,manner:6,mantl:13,manual:4,map:[0,3,14],map_plot:0,mapext:[0,14],mapplot:[0,14],mapview:7,march:[5,6],mass:[5,11],massdownload:6,master:13,mat:[0,4,7],match:[1,6],math:5,matlab:[0,4,7],matplotlib:[0,3,14],matrix:[0,4,6,14],matter:[0,19],max:[0,3,4],max_epid:6,maxdist:0,maxim:[0,3,4,5,7],maxima:0,maximis:6,maximum:[0,1,3,5],maxlat:[0,3,5,6],maxlon:[0,3,5,6],maxmim:6,maxz:[0,3],mc_backend:0,mean:[1,3,6],meant:[0,5,6],measur:[3,5],median:3,medium:3,memori:1,mention:7,merchant:3,merg:[0,3],mesh:[0,3],meshgrid:0,meter:0,method:[1,3,4,6,10,17,18,21],metric:0,mexican:5,middl:3,might:[0,6,11,17],migrat:[0,3,4,19],million:11,min:0,min_epid:6,mini:5,minillum:[0,14],minim:[6,7],minima:0,minimis:6,minimum:[0,6,7,13],minise:[0,1,5,11],minlat:[0,3,5,6],minlon:[0,3,5,6],minmag:[6,11],minu:3,minumum:11,minut:5,minz:0,miss:1,mistak:4,mit:3,mitig:7,mode:[0,1,3],model:[0,3,5,6,19],modifi:[0,1,3,4,5,6,18],modul:[0,1,3,4,21],mondai:[1,3,6],monoton:0,mont:3,more:[0,5,7,11,13,16,17,19],most:[0,5,11,17],mostli:4,mous:4,move:3,moveout:[2,19],moveout_test:4,mpi:[0,6],mpio:1,mpirun:0,mpl:0,mseed:[0,1,3,6,10,19],much:[0,5,7],multi:[0,5],multicor:4,multicore_stack:0,multipl:[0,3,5],multipli:0,multiprocess:[0,6],multit:[3,4,6],multit_con:4,multitap:[3,4,6],multitap_fqd:4,multitaper_weight:3,must:[1,5],mxn:0,myfil:[1,17],n_closest_point:0,name:[0,1,5,11],nameerror:6,nan:0,narrow:3,natur:5,nbin:3,ncode:1,ndarrai:[0,3,4,5],ndt:3,nearest:14,necessari:[0,3],need:[0,6],neg:0,neighbour:[0,14],nep06:1,net:5,netrestr:6,network:[0,1,3,4,5,6,7,11,17],never:0,newer:6,next:[0,5],nextpowerof2:5,nextpowof2:2,nez:[4,6],nez_fil:4,nois:[0,1,3,4,5,7,11],noisemat:6,nomatchingresponsehandl:6,non:[0,1],none:[0,1,3,4,5,6,11,14],noninfring:3,normal:3,north:4,note:[0,1,5],notebook:[15,16,19],notic:[3,11],notimplementederror:6,novemb:[0,3,5],now:[3,6,7,18],npz:[0,7],number:[0,1,3,4,5,20],numpi:[0,3,4,5,7,14],nx2:0,obj2stat:3,object:[0,1,3,4,5,6,8,11,14,17,18,20,21],obspi:[1,3,5,6,11,13,18],obtain:[3,20],occur:6,octob:[0,1,3,4,5,6],offer:6,often:1,old:[0,4,5,6],older:7,omega_min:3,onc:[1,11],one:[0,1,3,4,5,6,7,11,16,18,19],oneha:7,ones:[1,6,11],onli:[0,1,3,4,5,6,7,11,16,17,18],onset:[3,6],open:[0,1,11],optim:3,option:[0,1,3,4,5,6,11],order:[0,1,3,7],orfeu:5,org:[0,1,3,4,5,6],origin:[1,5,17],orign:6,orthogon:6,other:[0,1,3,4,5],otherwis:[1,3],ouput:4,our:[7,10],out:[0,3,16],outdat:3,outdir:4,outfil:4,outfold:1,outlin:[0,3],output:[0,1,3,4,5,6,16,18],outputdir:3,outputfil:[0,3],outputloc:6,outsid:0,over:[1,5,6,11,18],overlap:0,overload:11,overview:[1,3,18,21],overwrit:[0,5],own:17,p_direct:0,packag:[5,21],page:[1,21],param:[3,5,11],paramet:[0,1,3,4,5,6,7,10,19],parametr:5,paraview:0,parent:[0,3,4,5,6],part:[3,19],particular:[3,5],particularli:1,pass:[0,1,5],pat:1,patch:0,path:[1,3,4,11,17],pathname_or_url:3,pattern:[0,1],pdf:[0,3],peak:3,penal:1,per:[0,1,3,4,5,7,11,19],percentag:[3,6],perfect:0,perfom:1,perform:[1,3],period:7,permiss:3,permit:3,persist:1,person:3,peter:[0,1,3,4,5,6,13],petermaku:12,pfile:11,phase:[0,1,3,4,5,6,11,17],phasepick:0,pick:0,pick_phas:0,picker:0,pickl:[0,4,5,7],pierc:[0,3,7,9,19],pixel:7,pkl:[0,7,14],pkp:11,place:6,plain:[15,21],plan:11,pleas:19,plot:[0,3,7,19,21],plot_amplitud:0,plot_bin:0,plot_cross_sect:[2,14],plot_distribut:[3,19],plot_illum:0,plot_line_buff:2,plot_sect:16,plot_single_rf:16,plot_stat:0,plot_util:[2,14,16],plot_volume_sect:[0,14],plt:14,pof:5,point:[0,3,5,7,8,14,19,21],pol:[0,1,3,6,11,17],polar:[0,3],polaris:[1,3,6,11],polarisationfor:6,pole:[0,3],polici:1,polygon:0,popul:[0,7],portion:3,posit:[0,4],possibl:[0,6,17],postprocess:17,potenti:[7,11],potsdam:[0,1,3,4,5,6],power:[5,7,13],ppoint:[3,19],pps:0,pre:[3,6],predefin:[0,7],preempt:1,preemption:1,prepar:0,prepro_fold:6,prepro_subdir:[6,11],preprocess:[0,2,4,11,21],preprocessh5:2,preproloc:[0,4,6],preserv:0,previou:5,prf:[0,3,4,6],primari:[3,6],prime:1,primit:1,princeton:[0,5],print:[1,17],prior:[0,6],priori:5,probabl:[11,17],problem:0,problemat:0,process:[0,1,6,13,17],produc:[0,6],product:3,profil:[0,3],program:[1,4,7],proj_dir:[6,11],project:[3,6,11,13,18],properti:[1,3],provid:[0,1,3,4,5,6,7,9,11,14,16,17,19],pss:[0,3,4,6,11],pss_file:4,psvsh:6,publish:3,puls:4,purpos:[3,5,12],put:[0,5,15],pws:[0,3],pyasdf:[6,11],pyglim:[2,8,9,11,12,14,16,17],pyplot:[3,14],pyyaml:11,q_a:3,q_b:3,qcp:[4,6],qcs:[4,6],qlat:0,qlon:0,qrf:3,qualiti:[4,6],queri:[0,5],query_bin_tre:0,query_station_tre:0,question:3,quick:3,radial:6,radian:5,radii:7,radiu:[0,7],rai:[0,3,6,19],rais:[0,1,3,4,5,6],ram:[0,3,5],random:[4,5,16],rang:0,rate:[1,2,5],rather:[3,6],ratio:[0,4,11],raw:[0,1,5,6,11,13],raw_subdir:[6,11],rawdir:[1,11],rawfold:1,rawloc:[5,6],rawrf:0,rayp:[0,3,6],rayparamet:3,raypv:5,raysum:[3,4,5],raytrac:[0,3],raytrayc:[0,3],rdcc_nbyte:1,rdcc_nslot:1,rdcc_w0:1,reach:7,read:[0,1,3,4,7,11,14,16,18,19,21],read_by_st:3,read_ccp:[0,14],read_geom:4,read_hdf5_head:1,read_raysum:4,read_rf:[3,16,20],read_rfs_mat:4,readili:5,readonli:1,realli:[3,5],reason:[0,5,17],receciv:17,receiv:[0,1,3,4,6,7,8,9,11,13,15,18,19],reciproc:0,reckon:5,recommend:[0,1,6,7],recomput:0,record:[1,5,13,17],rectangl:0,rectangular:3,recurs:1,redon:3,redownload:[1,5,6],redownload_missing_statxml:1,redownload_statxml:6,reduc:1,refer:[3,9],refin:0,reflect:4,reftim:3,region:[6,9],regul:3,regular:3,regularis:3,regulis:3,reject:6,rel:[3,11],reli:11,remain:[0,3],remin:4,remov:[1,6,7,16],replac:3,repositori:[11,21],repres:6,request:[1,2,5,10,21],requir:[0,1],resampl:[1,3,5],resample_or_decim:5,resolut:[0,3,7],resolv:7,resons:4,resort:5,respect:[3,4,11,14],respons:[1,3,4,5,6,11],rest:5,restrict:[0,3,6],result:[0,1,5,7,16],ret:4,retain:4,retri:6,retriev:17,reveal:7,review:[4,13],rewrit:1,rewrite_to_hdf5:1,rf2:3,rf_mo:[3,4],rf_subdir:[6,11],rf_test:4,rf_with_my_funny_processing_idea:17,rfcmap:0,rfdatabas:[1,17],rfdb:[1,17],rfdir:3,rfh5:[2,17],rfloc:[0,4,6],rflogger:6,rfs:[0,4,18],rfst:[16,17],rfstack:1,rfstat:3,rfstream:[1,3,16,17,18,20,21],rftrace:[1,3,4,17,18,21],ricker:[4,5],right:[0,3,6,16],rigid:6,robust:0,rondenai:3,root:1,ros3:1,rose:19,rot:[6,11],rotat:[0,2,5,11],rotate_lqt_min:6,rotate_psv:6,rough:3,round:[5,7],roundhalf:2,routin:[14,15,21],row:6,rtz:[4,6,11],rtz_file:4,rule:1,run:3,s_f:5,sac:[0,3,4,6,10,16,18,19,21],safe:1,said:0,same:[0,1,3,5,6,11,15,19],sampl:[0,1,3,4,5,6],sampling_f:6,sampling_rate_new:5,saturdai:[5,6],save:[0,1,3,4,5,6,7,11,14,16,17,18,21],save_raw:5,save_raw_mse:5,saveasdf:[5,6],savefil:3,sawad:[0,5],scale:[3,7,11],scalingfactor:[3,16],scatter:13,script:4,search:[0,1,21],sec2:1,second:[3,5,6,11,16],section:[0,1,3,5,9,15,19],see:[0,1,3,6,11,17,19],seealso:6,seed:[5,17],seen:13,segment:5,seismic:[0,1,4,6],seismogram:[4,6,11],select:[0,1,5],self:[0,3,6],sell:3,sens:[4,6],seri:4,serv:[4,6],server:[6,11],set:[0,1,3,5,6,10,14],set_mpl_param:[14,16],sever:[0,1,6,18],shall:3,shape:[5,7],shearer:3,shelf:6,shelv:[6,11],shift:[3,4,5],shift_max:5,shortcut:3,shorten:3,shorter:[0,11],should:[0,1,3,5,6,7,11,17],show:[0,1,3,4,14,19],shown:[3,4,17],side:3,sigma:5,signal:[3,4,5,11],signalproc:2,significantli:7,simpl:[0,3,5,12,14,19],simplemodel:3,simplest:3,simpli:[0,14,16],simul:6,sinc:[3,11],singl:[1,3,6,15],size:[0,1,3,5],slat:6,slice2:3,slice:[0,3,14],slider:0,slightli:1,slon:6,slot:1,slow:[3,4,5,6],slower:5,slownesss:6,slst:5,small:[6,7,11],smaller:[5,7,11],smallest:1,snippet:14,snr:[4,6],snr_criteria:6,snrerror:6,softwar:3,some:[0,1,3,11,16,17],someth:3,sometim:7,sort:[4,5],sort_rat:4,sourc:[0,1,3,4,5,6,12],space:[0,1,5,11,14],spatial:[7,9],specif:[3,11,19],specifi:[1,6],spectral:[3,4,6,11],spectraldivis:[3,4],speed:7,sphere:[0,14],spheric:[0,3],split:3,squar:0,srf:[3,4,6],sshift:5,st_in:3,stack:[0,3,4,8,15,17,19,21],standard:[0,3,4,5,7,17],start:[0,3,6,15,18],starttim:[3,5,6,11],stastack:3,stat:[1,3,5],statfil:6,statio:1,station:[0,2,3,4,5,6,7,11,16,17,19],station_stack:[3,19],station_tre:0,stationxml:[1,5],statist:3,statlat:6,statloc:[0,1,5,6],statloc_subdir:[6,11],statlon:6,statrestr:6,statxml:[1,11],std:3,stdio:1,stdv:4,step:[10,11],stetp:3,still:[1,6,15],store:[6,7,18,19,21],str:[0,1,3,4,5,6],strategi:1,stream:[0,1,3,4,5,6,16,19],streamlengtherror:6,strictli:1,string:[0,3,4,5,6,11],strong:7,strongest:0,strongli:3,structur:[11,17],sts:4,style:[6,11],subdir:11,subfold:[5,6],subject:3,sublicens:3,submodel:5,subsequ:[0,3,6,14],subset:17,substanti:3,success:5,successor:13,suggest:11,suit:6,summari:21,sundai:5,supercrit:3,support:[1,3],suppos:[3,17,18],sure:[3,11],surfac:[3,6,7],swmr:1,synthet:2,system:[3,4,6,11],tabl:3,tag:[1,18],tailor:7,take:[0,1,3,4,5,7,14],taken:[0,3],taper:[3,4,6],taper_perc:6,taper_typ:6,tau:6,taup:6,taupi:3,taupymodel:6,team:[0,1,3,4,5,6],technic:0,techniqu:[3,13],teleseism:[0,1,5,6,11,13],templat:0,tend:[1,11],termin:0,test:[2,3,21],test_snr:4,test_tt_calcul:4,text:5,than:[0,1,3,5,6,7,11],thei:[0,3,4],them:[3,4,5,7,11,16,18,19],theoret:6,theoreth:[4,6],therefor:7,thesi:13,thi:[0,1,3,5,6,7,9,11,13,14,16,17,19,21],thick:3,thickness:3,thing:3,those:[0,7,9,11,15],though:7,thread:5,three:[0,3,4,6],threshold:7,through:[1,5,14],thu:5,thumb:1,thursdai:[0,1,3,4,5,6],time:[0,1,3,4,5,6,7,10,11,16,17,19],timelimit:16,timeshift:5,tini:15,titl:[0,3],tlim:16,tmp:6,tmpp:3,togeth:[3,11],tom:[3,18],tomographi:6,too:[0,3,6],tool:[0,11,14,15,21],toolset:[0,3],tort:3,total:1,total_dist:5,trace:[1,3,4,6,16,19],track:1,track_ord:1,tradeoff:7,transvers:[6,11],travel:6,treat:1,trim2:3,trim:3,truncat:[1,3],tshift:3,tt_model:3,tue:[5,6],tuesdai:[3,4,5,6],tupl:[0,1,3,4,5,6],tupleor:0,tutori:19,twice:11,two:[0,3,4,5,7,11],type:[0,1,3,4,5,6,17,19],typeerror:[0,1],uknown:[3,6],unavail:3,under:[0,1],understand:7,uneven:5,unexpect:17,union:0,uniqu:6,uniti:3,unix:[0,6,11],unknown:[0,3],unprocess:11,unstabl:5,unstack:1,unstructuredgrid:0,until:[0,3],unus:[0,3],updat:[5,6,7,11],upon:[0,18],upper:[0,3,5,13],uppermost:1,upscal:7,use:[0,1,3,4,5,6,7,11,15,17,19],usecas:17,used:[0,1,3,4,5,6,7,11,17],useful:[1,17],user:[0,1,3,11,13,17],userblock_s:1,uses:17,using:[0,1,3,5,6,7,9,11,14,16,18,19,20],usual:[0,3,5,6,7],utc:[3,6],utcdatetim:[1,3,5,6],util:[2,21],utilis:11,v108:1,v110:1,v112:1,v_nois:4,valid:11,valu:[0,1,3,6,7],valueerror:[0,5],var_rf:3,vari:9,variabl:[0,3,6],varianc:3,variant:11,variou:[3,4,5,6],vdep:0,vector:[0,3,4,5],ved:3,vel_model:0,veloc:[0,3,5,6,19],verbos:[0,1,6],veri:[3,5,6],version:[0,1,3,4,5,6,18],vertic:[3,6],vfd:1,via:[0,6],view:[0,16],visibl:0,visual:14,vlat:0,vlon:0,vmax:[0,14],vmin:[0,14],vmodel:[0,3],vmodel_fil:3,volum:[0,9,15],volumeexplor:0,volumeplot:[0,14],vpf:3,vplot:14,vsf:3,vtk:0,wai:[0,3,5,7,11],walk:[1,17],want:[3,5,6,7,11,16,17],warn:[1,6,11],warranti:3,wasn:6,wat:[3,4],water:[0,7],waterlevel:[3,4,6,11],wav_in_asdf:6,wav_in_db:6,wave:[3,5,6,13],waveform:[0,1,2,3,4,5,11,13,16,17,21],waveform_cli:[6,11],wavelet:[3,4,5,6],waypoin:0,waypoint:[0,5,14],webservic:[6,10],wed:4,wednesdai:[1,3,4,6],weight:[0,1,3,6,14],weigth:0,weird:15,well:[1,6,16],were:[4,5,18,21],what:7,when:[0,1,3,6,11],where:[1,6],wherea:[7,11],whether:[0,1,3,6,7,19],which:[0,1,3,4,5,6,7,11,16,17,20],whole:[0,5,7],whom:3,wide:11,width:[3,4,5],wildcard:[0,1,5,6,11,17],window:[0,3,5,6,10],wise:6,wish:[0,6,11],without:[3,14,19],work:[0,3,5,6,17,19],workflow:[4,6],world:11,would:[0,7],write:[0,1,3,4,6,18,19,21],write_info:6,write_st:1,writeraw:1,written:[0,1],wron:4,wrong:[0,1,3],www:[0,1,3,4,5,6],x_delta:3,xml:[1,5,6,11],xyz:5,yaml:11,yes:3,yet:7,yield:[1,4,5],yml:12,you:[0,1,3,5,6,7,11,16,17,18,19],your:[0,1,5,6,7,8,10,17,19,21],z_multipl:0,z_re:3,zero:5,zhu:[0,3],zmax:[0,14],zmin:0,zsl:[0,14]},titles:["pyglimer.ccp","pyglimer.database","API","pyglimer.rf","pyglimer.test","pyglimer.utils","pyglimer.waveform","CCPStack objects","CCP","Common Conversion Point Stacking","Receiver Functions","The Request Class","Installation","Introduction","CCP Stack Plotting","Plotting","Plotting Plain Receiver Functions","Receiver functions stored in hdf5 format","RF Handling","Methods available for both RFStream and RFTrace objects","Reading receiver functions that were saved in sac format","PyGLImER - A workflow to create a global database for Ps and Sp receiver functions in Python"],titleterms:{"class":11,"function":[10,16,17,20,21],The:11,api:2,asdf:1,avail:[17,19],bin:[0,7],both:19,ccp:[0,7,8,14],ccpstack:7,common:9,comput:0,convers:9,creat:[3,7,21],create_geom:5,createvmodel:5,cross:14,data:17,databas:[1,11,21],deconvolv:3,download:6,errorhandl:6,format:[17,20],geo_util:5,get:17,global:21,handl:18,hdf5:[11,17],indic:21,instal:12,introduct:13,line_buff:0,method:[11,19],moveout:3,mseed:11,nextpowof2:5,object:[7,19],over:17,overview:17,paramet:11,plain:16,plot:[14,15,16],plot_cross_sect:0,plot_line_buff:0,plot_util:0,point:9,preprocess:6,preprocessh5:6,pyglim:[0,1,3,4,5,6,7,21],python:21,rate:4,read:[17,20],receiv:[10,16,17,20,21],request:[6,11],rfh5:1,rfstream:19,rftrace:19,rotat:6,roundhalf:5,sac:[11,20],save:20,section:[14,16],set:11,signalproc:5,singl:16,stack:[7,9,14],station:1,store:17,synthet:4,tabl:21,tag:17,test:4,util:5,volum:14,waveform:6,were:20,workflow:21,write:17,your:11}})