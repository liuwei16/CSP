function dbEval
% Evaluate and plot all pedestrian detection results.

% -------------------------------------------------------------------------
% remaining parameters and constants
dataName = 'UsaTest';
annVer='new';             % annotation version('ori' or 'new')
fppiRn=[-2 0];            % fppi range for computing miss rate
aspectRatio = .41;        % default aspect ratio for all bbs
bnds = [5 5 635 475];     % discard bbs outside this pixel range
% if strcmp(annVer,'new'), bnds = [-inf -inf inf inf]; end 
plotRoc = 1;              % if true plot ROC else PR curves
plotAlg = 0;              % if true one plot per alg else one plot per exp
plotNum = 15;             % only show best plotNum curves (and VJ and HOG)
% samples = 10.^(fppiRn(1):.25:fppiRn(2)); % samples for computing area under the curve
samples = 10.^(-2:.25:0);
lims = [2e-4 50 .035 1];  % axis limits for ROC plots
% lims = [2e-4 50 .025 1];  % axis limits for ROC plots
bbsShow = 0;              % if true displays sample bbs for each alg/exp
bbsType = 'fp';           % type of bbs to display (fp/tp/fn/dt)

% -------------------------------------------------------------------------
% paths
pth.rootDir='./';
pth.resDir='../output/valresults/caltech/h/off';
pth.evalDir=[pth.rootDir 'ResultsEval/eval-' annVer];
pth.dtDir=[pth.rootDir 'ResultsEval/dt-'];
pth.gtDir=[pth.rootDir 'ResultsEval/gt-' annVer];
% pth.videoDir=[pth.rootDir 'data-USA/'];
% pth.annDir=[pth.rootDir 'anno_test_new'];

exps = {
  'Reasonable',     [50 inf],  [.65 inf], 0,   .5,  1.25
  'All',            [20 inf],  [.2 inf],  0,   .5,  1.25
  'Scale=large',    [100 inf], [inf inf], 0,   .5,  1.25
  'Scale=near',     [80 inf],  [inf inf], 0,   .5,  1.25
  'Scale=medium',   [30 80],   [inf inf], 0,   .5,  1.25
  'Scale=far',      [20 30],   [inf inf], 0,   .5,  1.25
  'Occ=none',       [50 inf],  [inf inf], 0,   .5,  1.25
  'Occ=partial',    [50 inf],  [.65 1],   0,   .5,  1.25
  'Occ=heavy',      [50 inf],  [.2 .65],  0,   .5,  1.25
  'Ar=all',         [50 inf],  [inf inf], 0,   .5,  1.25
  'Ar=typical',     [50 inf],  [inf inf],  .1, .5,  1.25
  'Ar=atypical',    [50 inf],  [inf inf], -.1, .5,  1.25
  'Overlap=25',     [50 inf],  [.65 inf], 0,   .25, 1.25
  'Overlap=50',     [50 inf],  [.65 inf], 0,   .50, 1.25
  'Overlap=75',     [50 inf],  [.65 inf], 0,   .75, 1.25
  'Expand=100',     [50 inf],  [.65 inf], 0,   .5,  1.00
  'Expand=125',     [50 inf],  [.65 inf], 0,   .5,  1.25
  'Expand=150',     [50 inf],  [.65 inf], 0,   .5,  1.50 };
exps=cell2struct(exps',{'name','hr','vr','ar','overlap','filter'});

% -------------------------------------------------------------------------
% List of algorithms: { name, resize, color, style }
%  name     - algorithm name (defines data location)
%  resize   - if true rescale height of each box by 100/128
n=300; clrs=zeros(n,3);
for i=1:n, clrs(i,:)=max(.3,mod([78 121 42]*(i+1),255)/255); end
alg_eval = dir(fullfile(pth.resDir));
algname = {alg_eval.name};
algsOt = cell(length(algname)-2,4);
for i=1:length(algname)-2
   algsOt{i,1} = algname{i+2};
   algsOt{i,2} = 0;
   algsOt{i,3} = clrs(i,:);
   if mod(i,2)==0
        algsOt{i,4} = '--';
   else
        algsOt{i,4} = '-';
   end
end

algsOt=cell2struct(algsOt',{'name','resize','color','style'});
for i=1:numel(algsOt), algsOt(i).type='other'; end
exps = exps(1);
algs =[algsOt(:)];    % select one or more algorithms for evaluation
% directory path
for i=1:numel(algs)
  algs(i).resDir = [pth.resDir '/' algs(i).name];
%   algs(i).dtDir  = [pth.dtDir algs(i).type '/'];
end

% select algorithms with results for current dataset
algs0=algs; names={algs0.name}; n=length(names); keep=false(1,n);
for i=1:n, keep(i)=exist([algs(i).resDir '/set06'],'dir'); end
algs=algs0(keep);

% name for all plots (and also temp directory for results)  
if(~exist(pth.evalDir,'dir')), mkdir(pth.evalDir); end

% load vbb all
% AS = loadAllVbb( pth.videoDir, dataName );
load('AS.mat');  
% load detections and ground truth and evaluate
dts = loadDt( algs, pth, aspectRatio, dataName, AS );
gts = loadGt( exps, pth, aspectRatio, bnds, dataName,  AS );
res = evalAlgs( pth.evalDir , algs, exps, gts, dts );
% compute the scores
[nGt,nDt]=size(res); xs=cell(nGt,nDt); ys=xs; scores=zeros(nGt,nDt);
for g=1:nGt
  for d=1:nDt
    [xs{g,d},ys{g,d},~,score] = ...
      bbGt('compRoc',res(g,d).gtr,res(g,d).dtr,plotRoc,samples);
    if(plotRoc), ys{g,d}=1-ys{g,d}; score=1-score; end
    if(plotRoc), score=exp(mean(log(score))); else score=mean(score); end
    scores(g,d)=score;
  end
end
% fName=[pth.evalDir '/Roc/'];
fName=pth.evalDir;
if(~exist(fName,'dir')), mkdir(fName); end
stra={res(1,:).stra}; stre={res(:,1).stre}; scores1=scores*100;%round(scores*10000)/100;
fName1 = [fName stre{1}];
f=fopen([fName1 '.txt'],'w');
for d=1:nDt, fprintf(f,'%s %f\n',stra{d},scores1(1,d)); end; fclose(f);
% plot curves and bbs
% plotExps( res, plotRoc, plotAlg, plotNum, pth.evalDir, ...
%           samples, lims, reshape([algs.color]',3,[])', {algs.style}, {algs.type} );
% 
% set(gcf, 'PaperPositionMode', 'manual');
% set(gcf, 'PaperUnits', 'inches');
% set(gcf, 'PaperPosition', [2.5 2.5 8.5 4])        
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function res = evalAlgs( plotName, algs, exps, gts, dts )
% Evaluate every algorithm on each experiment
%
% OUTPUTS
%  res    - nGt x nDt cell of all evaluations, each with fields
%   .stra   - string identifying algorithm
%   .stre   - string identifying experiment
%   .gtr    - [n x 1] gt result bbs for each frame [x y w h match]
%   .dtr    - [n x 1] dt result bbs for each frame [x y w h score match]
fprintf('Evaluating: %s\n',plotName); nGt=length(gts); nDt=length(dts);
res=repmat(struct('stra',[],'stre',[],'gtr',[],'dtr',[]),nGt,nDt);
for g=1:nGt
  for d=1:nDt
    gt=gts{g}; dt=dts{d}; n=length(gt); assert(length(dt)==n);
    stra=algs(d).name; stre=exps(g).name;
    fName = [plotName '/ev-' [stre '-' stra] '.mat'];
    if(exist(fName,'file')), R=load(fName); res(g,d)=R.R; continue; end
    fprintf('\tExp %i/%i, Alg %i/%i: %s/%s\n',g,nGt,d,nDt,stre,stra);
    hr = exps(g).hr.*[1/exps(g).filter exps(g).filter];
    for f=1:n, bb=dt{f}; dt{f}=bb(bb(:,4)>=hr(1) & bb(:,4)<hr(2),:); end
    [gtr,dtr] = bbGt('evalRes',gt,dt,exps(g).overlap);
    R=struct('stra',stra,'stre',stre,'gtr',{gtr},'dtr',{dtr});
    res(g,d)=R; save(fName,'R');
  end
end
end

% -------------------------------------------------------------------------
function plotExps( res, plotRoc, plotAlg, plotNum, plotName, ...
  samples, lims, colors, styles, algtypes )
% Plot all ROC or PR curves.
%
% INPUTS
%  res      - output of evalAlgs
%  plotRoc  - if true plot ROC else PR curves
%  plotAlg  - if true one plot per alg else one plot per exp
%  plotNum  - only show best plotNum curves (and VJ and HOG)
%  plotName - filename for saving plots
%  samples  - samples for computing area under the curve
%  lims     - axis limits for ROC plots
%  colors   - algorithm plot colors
%  styles   - algorithm plot linestyles

% Compute (xs,ys) and score (area under the curve) for every exp/alg
[nGt,nDt]=size(res); xs=cell(nGt,nDt); ys=xs; scores=zeros(nGt,nDt);
for g=1:nGt
  for d=1:nDt
    [xs{g,d},ys{g,d},~,score] = ...
      bbGt('compRoc',res(g,d).gtr,res(g,d).dtr,plotRoc,samples);
    if(plotRoc), ys{g,d}=1-ys{g,d}; score=1-score; end
    if(plotRoc), score=exp(mean(log(score))); else score=mean(score); end
    scores(g,d)=score;
  end
end

% Generate plots
if( plotRoc )
  fName=[plotName '/Roc/'];
  if(~exist(fName,'dir')), mkdir(fName); end
else
  fName=[plotName '/Pr/']; 
  if(~exist(fName,'dir')), mkdir(fName); end
end
stra={res(1,:).stra}; stre={res(:,1).stre}; scores1=scores*100;%round(scores*10000)/100;
if( plotAlg ), nPlots=nDt; else nPlots=nGt; end; plotNum=min(plotNum,nDt);
for p=1:nPlots
  % prepare xs1,ys1,lgd1,colors1,styles1,fName1 according to plot type
  if( plotAlg )
    xs1=xs(:,p); ys1=ys(:,p); fName1=[fName stra{p}]; lgd1=stre;
    for g=1:nGt, lgd1{g}=sprintf('%.2i%% %s',scores1(g,p),stre{g}); end
    colors1=uniqueColors(1,max(10,nGt)); styles1=repmat({'-','--'},1,nGt);
  else
    xs1=xs(p,:); ys1=ys(p,:); fName1=[fName stre{p}]; lgd1=stra;
    for d=1:nDt, lgd1{d}=sprintf('%.2f%% %s',scores1(p,d),stra{d}); end
    kp=[find(strcmp(stra,'VJ')) find(strcmp(stra,'HOG')) 1 1];
    [~,ord]=sort(scores(p,:)); 
%     kp=ord==kp(1)|ord==kp(2); j=find(cumsum(~kp)>=plotNum-2); 
    kp=false(size(ord));      j=find(cumsum(~kp)>=plotNum-0);
    kp(1:j(1))=1; ord=fliplr(ord(kp));
    xs1=xs1(ord); ys1=ys1(ord); lgd1=lgd1(ord); colors1=colors(ord,:);
    styles1=styles(ord); f=fopen([fName1 '.txt'],'w');
    algtypes1=algtypes(ord);
    for d=1:nDt, fprintf(f,'%s %f\n',stra{d},scores(p,d)); end; fclose(f);
  end
  % plot curves and finalize display
  figure(1); clf; grid on; hold on; n=length(xs1); h=zeros(1,n);
  for i=1:n, h(i)=plot(xs1{i},ys1{i},'Color',colors1(i,:),...
      'LineStyle',styles1{i},'LineWidth',2); end
  if( plotRoc )
    yt=[.05 .1:.1:.5 .64 .8]; ytStr=int2str2(yt*100,2);
    for i=1:length(yt), ytStr{i}=['.' ytStr{i}]; end
    set(gca,'XScale','log','YScale','log',...
      'YTick',[yt 1],'YTickLabel',[ytStr '1'],...
      'XMinorGrid','off','XMinorTic','off',...
      'YMinorGrid','off','YMinorTic','off');
    xlabel('false positives per image','FontSize',14);
    ylabel('miss rate','FontSize',14); axis(lims);
  else
    x=1; for i=1:n, x=max(x,max(xs1{i})); end, x=min(x-mod(x,.1),1.0);
    y=.8; for i=1:n, y=min(y,min(ys1{i})); end, y=max(y-mod(y,.1),.01);
    xlim([0, x]); ylim([y, 1]); set(gca,'xtick',0:.1:1);
    xlabel('Recall','FontSize',14); ylabel('Precision','FontSize',14);
  end  
  for i=1:numel(lgd1)
    if strcmp(algtypes1{i},'my')
      lgd1{i} = ['\bf{' lgd1{i} '}'];
    end
  end
%   h1=legend(h,lgd1,'Location','sw','FontSize',11); legend(h1,'boxoff'); 
%   h1=legend(h,lgd1,'Location','SouthEastOutside','FontSize',11); legend(h1,'boxoff');  
  legend(h,lgd1,'Location','ne','FontSize',10);
  
  % save figure to disk (uncomment pdfcrop commands to automatically crop)
  savefig(fName1,1,'pdf','-r300','-fonts'); %close(1);
  if(0), setenv('PATH',[getenv('PATH') ':/usr/texbin/']); end
  if(0), system(['pdfcrop -margins ''-30 -20 -50 -10 '' ' ...
      fName1 '.pdf ' fName1 '.pdf']); end
end

end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AS = loadAllVbb( videoDir, dataName )
% Load given annotation (caches AS for speed).
[setIds,vidIds]=getDBInfo(dataName);
load()
AS=cell(length(setIds),1e3);
for s=1:length(setIds), s1=setIds(s); 
  for v=1:length(vidIds{s}), v1=vidIds{s}(v);
    fName=sprintf('%s/annotations/set%02i/V%03i',videoDir,s1,v1);
    A=vbb('vbbLoad',fName); AS{s,v}=A;
  end
end
save('AS.mat','AS','-v6');
end
% -------------------------------------------------------------------------
function gts = loadGt( exps, pth, aspectRatio, bnds, dataName, AS )
% Load ground truth of all experiments for all frames.
fprintf('Loading ground truth: %s\n',pth.gtDir);
nExp=length(exps); gts=cell(1,nExp);
[setIds,vidIds,skip] = getDBInfo(dataName);
if(~exist(pth.gtDir,'dir')), mkdir(pth.gtDir); end
for i=1:nExp
  gName = [pth.gtDir '/gt-' exps(i).name '.mat'];
  if(exist(gName,'file')), gt=load(gName); gts{i}=gt.gt; continue; end
  fprintf('\tExperiment #%d: %s\n', i, exps(i).name);
  gt=cell(1,100000); k=0; 
  lbls={'person','person?','people','ignore'};
%   lbls={'person'};
  ilbls={'ignore'};
 pLoad={'lbls', lbls, 'ilbls', ilbls,...
     'hRng',exps(i).hr,'vRng',exps(i).vr, ...
     'xRng',bnds([1 3]),'yRng',bnds([2 4])};
  for s=1:length(setIds), s1=setIds(s);
    for v=1:length(vidIds{s}), v1=vidIds{s}(v);
      A = AS{s,v};
      for f=skip-1:skip:A.nFrame-1
        annName=sprintf('%s/set%02d_V%03d_I%05d.txt', pth.annDir, s1, v1, f);
        [~,bb]=bbGt('bbLoad',annName,pLoad); ids=bb(:,5)~=1;
        bb(ids,:)=bbApply('resize',bb(ids,:),1,0,aspectRatio); 
        k=k+1; gt{k}=bb;
      end
    end
  end
  gt=gt(1:k); gts{i}=gt; save(gName,'gt','-v6');
end 
end

% -------------------------------------------------------------------------
function dts = loadDt( algs, pth, aspectRatio, dataName, AS )
% Load detections of all algorithm for all frames.
fprintf('Loading detections: %s\n',pth.dtDir);
nAlg=length(algs); dts=cell(1,nAlg);
[setIds,vidIds,skip] = getDBInfo(dataName);
alltype=unique({algs(:).type});
for i=1:numel(alltype)
  dirName=[pth.dtDir alltype{i}];
  if(~exist(dirName,'dir')), mkdir(dirName); end
end
for i=1:nAlg
  aName = [pth.dtDir algs(i).type '/dt-' algs(i).name '.mat'];
  if(exist(aName,'file')), dt=load(aName); dts{i}=dt.dt; continue; end
  fprintf('\tAlgorithm #%d: %s\n', i, algs(i).name);
  dt=cell(1,100000); k=0; aDir=algs(i).resDir;
  if(algs(i).resize), resize=100/128; else resize=1; end
  for s=1:length(setIds), s1=setIds(s);
    for v=1:length(vidIds{s}), v1=vidIds{s}(v);
      A=AS{s,v}; frames=skip-1:skip:A.nFrame-1;
      vName=sprintf('%s/set%02d/V%03d',aDir,s1,v1);   
      if(~exist([vName '.txt'],'file'))
        % consolidate bbs for video into single text file
        bbs=cell(length(frames),1);
        for f=1:length(frames)
          fName = sprintf('%s/I%05d.txt',vName,frames(f));
          if(~exist(fName,'file')), error(['file not found:' fName]); end
          bb=load(fName,'-ascii'); if(isempty(bb)), bb=zeros(0,5); end
          if(size(bb,2)~=5), error('incorrect dimensions'); end
          bbs{f}=[ones(size(bb,1),1)*(frames(f)+1) bb];
        end
        for f=frames, delete(sprintf('%s/I%05d.txt',vName,f)); end
        bbs=cell2mat(bbs); dlmwrite([vName '.txt'],bbs); rmdir(vName,'s');
      end
      bbs=load([vName '.txt'],'-ascii');      
      if isempty(bbs), bbs=zeros(1,6); end    
      for f=frames, bb=bbs(bbs(:,1)==f+1,2:6);
%         bb=bbApply('resize',bb,resize,0,aspectRatio);
        k=k+1; dt{k}=bb;
      end         
    end
  end
  dt=dt(1:k); dts{i}=dt; save(aName,'dt','-v6');
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [setIds,vidIds,skip,ext]=getDBInfo(name1)
vidId=[]; setId=[];
switch lower(name1)
  case 'usatrain' % Caltech Pedestrian Datasets (training)
    setIds=0:5; skip=30; ext='jpg';
    vidIds={0:14 0:5 0:11 0:12 0:11 0:12};
  case 'usatest' % Caltech Pedestrian Datasets (testing)
    setIds=6:10; skip=30; ext='jpg';
    vidIds={0:18 0:11 0:10 0:11 0:11};
  otherwise, error('unknown data type: %s',name);
end

% optionally select only specific set/vid if name ended in ints
if(~isempty(setId)), setIds=setIds(setId); vidIds=vidIds(setId); end
if(~isempty(vidId)), vidIds={vidIds{1}(vidId)}; end

end