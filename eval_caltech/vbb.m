function varargout = vbb( action, varargin )
% Data structure for video bounding box (vbb) annotations.
%
% A video bounding box (vbb) annotation stores bounding boxes (bbs) for
% objects of interest. The primary difference from a static annotation is
% that each object can exist for multiple frames, ie, a vbb annotation not
% only provides the locations of objects but also tracking information. A
% vbb annotation A is simply a Matlab struct. It contains data per object
% (such as a string label) and data per object per frame (such as a bb).
% Each object is identified with a unique integer id.
%
% Data per object (indexed by integer id) includes the following fields:
%  init - 0/1 value indicating whether object w given id exists
%  lbl  - a string label describing object type (eg: 'pedestrian')
%  str  - the first frame in which object appears (1 indexed)
%  end  - the last frame in which object appears (1 indexed)
%  hide - 0/1 value indicating object is 'hidden' (used during labeling)
%
% Data per object per frame (indexed by frame and id) includes:
%  pos  - [l t w h]: bb indicating predicted object extent
%  posv - [l t w h]: bb indicating visible region (may be [0 0 0 0])
%  occl - 0/1 value indicating if bb is occluded
%  lock - 0/1 value indicating bb is 'locked' (used during labeling)
%
% vbb contains a number of utility functions for working with an
% annotation A, making it generally unnecessary to access the fields of A
% directly. The format for accessing the various utility functions is:
%  outputs = vbb( 'action', inputs );
% Below is a list of utility functions, broken up into 3 categories.
% Occasionally more help is available via a call to help "vbb>action".
%
% %%% init and save/load annotation to/from disk
% Create new annotation for given length video
%   A = vbb( 'init', nFrame, maxObj )
% Generate annotation filename (add .vbb and optionally time stamp)
%   [fName,ext] = vbb( 'vbbName', fName, [timeStmp], [ext] )
% Save annotation A to fName with optional time stamp (F by default)
%   vbb('vbbSave', A, fName, [timeStmp] )
% Load annotation from disk:
%   A = vbb('vbbLoad', fName )
% Save annotation A to fName (in .txt format):
%   vbb('vbbSaveTxt', A, fName, timeStmp )
% Load annotation from disk (in .txt format):
%   A = vbb('vbbLoadTxt', fName )
% Export single frame annotations to tarDir/*.txt
%   vbb( 'vbbToFiles', A, tarDir, [fs], [skip], [f0], [f1] )
% Combine single frame annotations from srcDir/*.txt
%   [A,fs] = vbb( 'vbbFrFiles', srcDir, [fs] )
%
% %%% inspect / alter annotation
% Get number of unique objects in annotation
%   n = vbb( 'numObj', A )
% Get an unused object id (for adding a new object)
%   [A,id] = vbb( 'newId', A )
% Create a new, empty object (not added to A)
%   [A,obj] = vbb( 'emptyObj', A, [frame] )
% Get struct with all data from frames s-e for given object
%   obj = vbb( 'get', A, id, [s], [e] )
% Add object to annotation
%   A = vbb( 'add', A, obj )
% Remove object from annotation
%   A = vbb( 'del', A, id )
% Crop or extend object temporally
%   A = vbb( 'setRng', A, id, s, e )
% Get object information, see above for valid properties
%   v = vbb( 'getVal', A, id, name, [frmS], [frmE] )
% Set object information, see above for valid properties
%   A = vbb( 'setVal', A, id, name, v, [frmS], [frmE] )
%
% %%% other functions
% Visulatization: draw annotation on top of current image
%   hs = vbb( 'drawToFrame', A, frame )
% Uses seqPlayer to display seq file with overlayed annotations.
%   vbb( 'vbbPlayer', A, srcName )
% Visulatization: create seq file w annotation superimposed.
%   vbb( 'drawToVideo', A, srcName, tarName )
% Shift entire annotation by del frames. (useful for synchronizing w video)
%   A = vbb( 'timeShift', A, del )
% Ensure posv is fully contained in pos
%   A = vbb( 'swapPosv', A, validate, swap, hide )
% Stats: get stats about annotation
%   [stats,stats1,logDur] = vbb( 'getStats', A );
% Returns the ground truth bbs for a single frame.
%   [gt,posv,lbls] = vbb( 'frameAnn', A, frame, lbls, test )
%
% USAGE
%  varargout = vbb( action, varargin );
%
% INPUTS
%  action     - string specifying action
%  varargin   - depends on action, see above
%
% OUTPUTS
%  varargout  - depends on action, see above
%
% EXAMPLE
%
% See also BBAPPLY
%
% Caltech Pedestrian Dataset     Version 3.2.1
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

%#ok<*DEFNU>
varargout = cell(1,nargout);
[varargout{:}] = eval([action '(varargin{:});']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% init / save / load

function A = init( nFrame, maxObj )

if( nargin<2 || isempty(maxObj) ), maxObj=16; end

A.nFrame   = nFrame;
A.objLists = cell(1,nFrame);
A.maxObj   = maxObj;
A.objInit  = zeros(1,A.maxObj);
A.objLbl   = cell(1,A.maxObj);
A.objStr   = -ones(1,A.maxObj);
A.objEnd   = -ones(1,A.maxObj);
A.objHide  = zeros(1,A.maxObj);

A.altered  = false;
A.log      = 0;
A.logLen   = 0;
end

function [fName,ext] = vbbName( fName, timeStmp, ext )
if(nargin<2), timeStmp=false; end; if(nargin<3), ext=''; end
[d,f,ext1]=fileparts(fName); if(isempty(ext)), ext=ext1; end
if(isempty(ext)), ext='.vbb'; end
if(timeStmp), f=[f '-' regexprep(datestr(now), ':| ', '-')]; end
if(isempty(d)), d='.'; end; fName=[d '/' f ext];
end

function vbbSave( A, fName, timeStmp )
if(nargin<3), timeStmp=false; end; vers=1.4; %#ok<NASGU>
[fName,ext]=vbbName(fName,timeStmp);
if(strcmp(ext,'.txt')), vbbSaveTxt(A,fName,timeStmp); return; end
A=cleanup(A); save(fName,'A','vers','-v6'); %#ok<NASGU>
end

function A = vbbLoad( fName )
[fName,ext]=vbbName(fName); vers=1.4;
if(strcmp(ext,'.txt')), A=vbbLoadTxt(fName); return; end
L = load( '-mat', fName );
if( ~isfield(L,'A') || ~isfield(L,'vers') );
  error('Not a valid video annoation file.');
end;
A = L.A;
% .06 -> 1.0 conversion (add log/logLen)
if( L.vers==.06 )
  L.vers=1.0; A.log=0; A.logLen=0;
end
% 1.0 -> 1.1 conversion (add trnc field)
if( L.vers==1.0 )
  L.vers=1.1;
  for f=1:A.nFrame
    if(isempty(A.objLists{f})); A.objLists{f}=[]; end
    for j=1:length(A.objLists{f}); A.objLists{f}(j).trnc=0; end
  end
end
% 1.1 -> 1.2 conversion (add hide/posv fields)
if( L.vers==1.1 )
  L.vers=1.2;
  for f=1:A.nFrame
    if(isempty(A.objLists{f})); A.objLists{f}=[]; end
    for j=1:length(A.objLists{f}); A.objLists{f}(j).posv=[0 0 0 0]; end
  end
  A.objHide = zeros(1,A.maxObj);
end
% 1.2 -> 1.3 conversion (remove trnc field)
if( L.vers==1.2 )
  L.vers=1.3;
  for f=1:A.nFrame
    if(isempty(A.objLists{f})); A.objLists{f}=[]; else
      A.objLists{f} = rmfield(A.objLists{f},'trnc');
    end
  end
end
% 1.3 -> 1.4 conversion (remove objAr field)
if( L.vers==1.3 )
  L.vers=1.4; A = rmfield(A,'objAr');
end
% check version
if( L.vers~=vers )
  er = [num2str(L.vers) ' (current: ' num2str(vers) ')'];
  error(['Incompatible versions: ' er]);
end
% order fields
order={'nFrame','objLists','maxObj','objInit','objLbl','objStr',...
  'objEnd','objHide','altered','log','logLen'};
A = orderfields(A,order);
A.altered = false;
end

function vbbSaveTxt( A, fName, timeStmp )
if(nargin<3), timeStmp=false; end; vers=1.4;
fName=vbbName(fName,timeStmp,'.txt');
A=cleanup(A,0); n=numObj(A); nFrame=A.nFrame;
fid=fopen(fName,'w'); assert(fid>0);
% write header info to text
fp=@(varargin) fprintf(fid,varargin{:});
fp('%% vbb version=%f\n',vers); fp('nFrame=%i n=%i\n',nFrame,n);
fp('log=['); fp('%f ',A.log); fp(']\n');
% write each object to text
for id=1:n, o=get(A,id);
  fp('\n-----------------------------------\n');
  fp('lbl=''%s'' str=%i end=%i hide=%i\n',o.lbl,o.str,o.end,o.hide);
  fp('pos =['); fp('%f %f %f %f; ',o.pos'); fp(']\n');
  fp('posv=['); fp('%f %f %f %f; ',o.posv'); fp(']\n');
  fp('occl=['); fp('%i ',o.occl); fp(']\n');
  fp('lock=['); fp('%i ',o.lock); fp(']\n');
end
fclose(fid);
end

function A = vbbLoadTxt( fName )
fName=vbbName(fName,0,'.txt'); vers=1.4;
if(~exist(fName,'file')), error([fName ' not found']); end
try
  % read in header and create A
  f=fopen(fName,'r'); s=fgetl(f); v=sscanf(s,'%% vbb version=%f');
  if(v~=vers), error('Incompatible versions: %f (current=%f)',v,vers); end
  s=fgetl(f); r=sscanf(s,'nFrame=%d n=%d'); nFrame=r(1); n=r(2);
  s=fgetl(f); assert(strcmp(s(1:5),'log=[')); assert(s(end)==']');
  log=sscanf(s(6:end-1),'%f ')'; A=init(nFrame,n);
  % read in each object in turn
  for id=1:n
    s=fgetl(f); assert(isempty(s));
    s=fgetl(f); assert(strcmp(s,'-----------------------------------'));
    s=fgetl(f); r=textscan(s,'lbl=%s str=%d end=%d hide=%d');
    [A,o]=emptyObj(A,0); o.lbl=r{1}{1}(2:end-1);
    o.str=r{2}; o.end=r{3}; o.hide=r{4};
    s=fgetl(f); assert(strcmp(s(1:6),'pos =[')); assert(s(end)==']');
    pos=sscanf(s(7:end-1),'%f %f %f %f;'); o.pos=reshape(pos,4,[])';
    s=fgetl(f); assert(strcmp(s(1:6),'posv=[')); assert(s(end)==']');
    posv=sscanf(s(7:end-1),'%f %f %f %f;'); o.posv=reshape(posv,4,[])';
    s=fgetl(f); assert(strcmp(s(1:6),'occl=[')); assert(s(end)==']');
    o.occl=sscanf(s(7:end-1),'%d ');
    s=fgetl(f); assert(strcmp(s(1:6),'lock=[')); assert(s(end)==']');
    o.lock=sscanf(s(7:end-1),'%d ');
    A=add(A,o);
  end
  if(isempty(log)), A.log=0; A.logLen=0; else
    A.log=log; A.logLen=length(log); end
  A.altered=false; fclose(f);
catch e, fclose(f); throw(e);
end
end

function vbbToFiles( A, tarDir, fs, skip, f0, f1 )
% export single frame annotations to tarDir/*.txt
nFrm=A.nFrame; fName=@(f) ['I' int2str2(f-1,5) '.txt'];
if(nargin<3 || isempty(fs)), for f=1:nFrm, fs{f}=fName(f); end; end
if(nargin<4 || isempty(skip)), skip=1; end
if(nargin<5 || isempty(f0)), f0=1; end
if(nargin<6 || isempty(f1)), f1=nFrm; end
if(~exist(tarDir,'dir')), mkdir(tarDir); end
for f=f0:skip:f1
  nObj=length(A.objLists{f}); objs=bbGt('create',nObj);
  for j=1:nObj
    o=A.objLists{f}(j); objs(j).lbl=A.objLbl{o.id}; objs(j).occ=o.occl;
    objs(j).bb=round(o.pos); objs(j).bbv=round(o.posv);
  end
  bbGt('bbSave',objs,[tarDir '/' fs{f}]);
end
end

function [A,fs] = vbbFrFiles( srcDir, fs )
% combine single frame annotations from srcDir/*.txt
if(nargin<2 || isempty(fs)), fs=dir([srcDir '/*.txt']); fs={fs.name}; end
nFrm=length(fs); A=init(nFrm);
for f=1:nFrm
  objs = bbGt('bbLoad',[srcDir '/' fs{f}]);
  for j=1:length(objs)
    [A,obj]=emptyObj(A,f); o=objs(j); obj.lbl=o.lbl;
    obj.pos=o.bb; obj.occl=o.occ; obj.posv=o.bbv; A=add(A,obj);
  end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inspect / alter annotation

function n = numObj( A )
n = sum( A.objInit );
end

function [A,id] = newId( A )
[val,id] = min( A.objInit );
if( isempty(val) || val~=0 );
  A = doubleLen( A );
  [val,id] = min( A.objInit );
end
assert(val==0);
end

function [A,obj] = emptyObj( A, frame )
[A,id] = newId( A );
obj.id   = id;
obj.lbl  = '';
obj.hide = 0;
if(nargin<2)
  obj.str  = -1;
  obj.end  = -1;
  len = 0;
else
  obj.str  = frame;
  obj.end  = frame;
  len = 1;
end
obj.pos  = zeros(len,4);
obj.posv = zeros(len,4);
obj.occl = zeros(len,1);
obj.lock = ones(len,1);
end

function obj = get( A, id, s, e )

assert( 0<id && id<=A.maxObj );
assert( A.objInit(id)==1 );

if(nargin<3); s=A.objStr(id); else assert(s>=A.objStr(id)); end;
if(nargin<4); e=A.objEnd(id); else assert(e<=A.objEnd(id)); end;

% get general object info
obj.id   = id;
obj.lbl  = A.objLbl{id};
obj.str  = s;
obj.end  = e;
obj.hide = A.objHide(id);

% get per-frame object info
len = obj.end-obj.str+1;
obj.pos  = zeros(len,4);
obj.posv = zeros(len,4);
obj.occl = zeros(len,1);
obj.lock = zeros(len,1);
for i=1:len
  f = obj.str+i-1;
  objList = A.objLists{f};
  obj1 = objList([objList.id]==id);
  obj.pos(i,:)  = obj1.pos;
  obj.posv(i,:) = obj1.posv;
  obj.occl(i)   = obj1.occl;
  obj.lock(i)   = obj1.lock;
end

end

function A = add( A, obj )

% check id or get new id
id = obj.id;
if( id==-1 )
  [A,id] = newId( A );
else
  assert( 0<id && id<=A.maxObj );
  assert( A.objInit(id)==0 );
end

% save general object info
A.objInit(id) = 1;
A.objLbl{id}  = obj.lbl;
A.objStr(id)  = obj.str;
A.objEnd(id)  = obj.end;
A.objHide(id) = obj.hide;

% save per-frame object info
len = obj.end - obj.str + 1;
assert( size(obj.pos,1)==len );
assert( size(obj.posv,1)==len );
assert( size(obj.occl,1)==len );
assert( size(obj.lock,1)==len );
for i = 1:len
  obj1.id   = id;
  obj1.pos  = obj.pos(i,:);
  obj1.posv = obj.posv(i,:);
  obj1.occl = obj.occl(i);
  obj1.lock = obj.lock(i);
  f = obj.str+i-1;
  A.objLists{f}=[A.objLists{f} obj1];
end

A = altered( A );
end

function A = del( A, id )

assert( 0<id && id<=A.maxObj );
assert( A.objInit(id)==1 );

% delete per-frame object info
objStr = A.objStr(id);
objEnd = A.objEnd(id);
len = objEnd-objStr+1;
for i=1:len
  f = objStr+i-1;
  objList = A.objLists{f};
  objList([objList.id]==id) = [];
  A.objLists{f} = objList;
end

% delete general object info
A.objInit(id) = 0;
A.objLbl{id}  = [];
A.objStr(id)  = -1;
A.objEnd(id)  = -1;
A.objHide(id) = 0;

A = altered( A );
end

function A = setRng( A, id, s, e )

assert( s>=1 && e<=A.nFrame && s<=e && A.objInit(id)==1 );
s0=A.objStr(id); e0=A.objEnd(id); assert( e>=s0 && e0>=s );
if(s==s0 && e==e0), return; end; A.objStr(id)=s; A.objEnd(id)=e;

if( s0>s )
  objs=A.objLists{s0}; obj=objs([objs.id]==id); obj.occl=0; obj.lock=0;
  for f=s:s0-1, A.objLists{f}=[A.objLists{f} obj]; end
elseif( s0<s )
  for f=s0:s-1, os=A.objLists{f}; os([os.id]==id)=[]; A.objLists{f}=os; end
end

if( e>e0 )
  objs=A.objLists{e0}; obj=objs([objs.id]==id); obj.occl=0; obj.lock=0;
  for f=e0+1:e, A.objLists{f}=[A.objLists{f} obj]; end
elseif( e<e0 )
  for f=e+1:e0, os=A.objLists{f}; os([os.id]==id)=[]; A.objLists{f}=os; end
end

A = altered( A );
end

function v = getVal( A, id, name, frmS, frmE )

if(nargin<4); frmS=[]; end;
if(nargin<5); frmE=frmS; end;
assert(strcmp(name,'init') || A.objInit(id)==1);
switch name
  case 'lbl'
    assert( isempty(frmS) );
    v = A.objLbl{id};
  case {'init','str','end','hide'}
    assert( isempty(frmS) );
    name = ['obj' upper(name(1)) name(2:end)];
    v = A.(name)(id);
  case {'pos','posv','occl','lock'}
    assert( ~isempty(frmS) );
    assert( A.objStr(id)<=frmS && frmE<=A.objEnd(id) );
    frms = frmS:frmE; len=length(frms);
    for f=1:len
      objList = A.objLists{frms(f)};
      v1 = objList([objList.id]==id).(name);
      if( f==1 ); v=repmat(v1,[len 1]); else v(f,:) = v1; end
    end
  otherwise
    error( ['invalid field: ' name] );
end

end

function A = setVal( A, id, name, v, frmS, frmE )

if(nargin<5); frmS=[]; end;
if(nargin<6); frmE=frmS; end;
assert( A.objInit(id)==1 );
switch name
  case 'lbl'
    assert( isempty(frmS) );
    A.objLbl{id} = v;
  case {'hide'}
    assert( isempty(frmS) );
    name = ['obj' upper(name(1)) name(2:end)];
    A.(name)(id) = v;
  case {'pos','posv','occl','lock'}
    assert( ~isempty(frmS) );
    assert( A.objStr(id)<=frmS && frmE<=A.objEnd(id) );
    frms = frmS:frmE; len=length(frms);
    for f=1:len
      objList = A.objLists{frms(f)};
      objList([objList.id]==id).(name) = v(f,:);
      A.objLists{frms(f)} = objList;
    end
  otherwise
    error( ['invalid/unalterable field: ' name] );
end

A = altered( A );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% other functions

function hs = drawToFrame( A, frame )
hs=[];
for o=A.objLists{frame}
  hr = bbApply('draw', o.pos, 'g', 2, '-' );
  if(all(o.posv)==0), hrv=[]; else
    hrv = bbApply('draw', o.posv, 'y', 2, '--' );
  end
  label = [A.objLbl{o.id} ' [' int2str(o.id) ']'];
  ht = text( o.pos(1), o.pos(2)-10, label );
  set( ht, 'color', 'w', 'FontSize', 10, 'FontWeight', 'bold' );
  hs = [hs hr ht hrv]; %#ok<AGROW>
end
end

function vbbPlayer( A, srcName )
dispFunc=@(f) vbb('drawToFrame',A,f+1);
seqPlayer(srcName,dispFunc);
end

function drawToVideo( A, srcName, tarName )
% open video to read and make video to write
assert( ~strcmp(srcName,tarName) );
sr=seqIo(srcName,'r'); info=sr.getinfo();
nFrame=info.numFrames; w=info.width; h=info.height;
assert(A.nFrame==nFrame); sw=seqIo(tarName,'w',info);
% display and write each frame
ticId=ticStatus; hf=figure; hAx=axes('parent',hf);
hIm=imshow(zeros(h,w,3,'uint8'),'parent',hAx); truesize;

for i=1:nFrame
  I=sr.getnext(); set(hIm,'CData',I); hs=drawToFrame(A,i);
  I=getframe; I=I.cdata; I=I(1:h,1:w,:);
  sw.addframe(I); delete(hs); tocStatus( ticId, i/nFrame );
end
sr.close(); sw.close(); close(hf);
end

function A = timeShift( A, del )
% shift entire annotation by del frames
nFrame=A.nFrame; locs=logical(A.objInit);
if(del>0), is=locs & A.objStr==1; A.objStr(is)=A.objStr(is)-del; end
A.objStr(locs)=min(max(A.objStr(locs)+del,1),nFrame);
A.objEnd(locs)=min(max(A.objEnd(locs)+del,1),nFrame);
if( del>0 ) % replicate annotation for first frame del times
  A.objLists=[A.objLists(ones(1,del)) A.objLists(1:end-del)];
else % no annotations for last del frames
  A.objLists=[A.objLists(1-del:end) cell(1,-del)];
end
end

function A = swapPosv( A, validate, swap, hide )
% Swap pos/posv and ensure pos/posv consistent.
%
% The visible region of an object (posv) should be a subset of the
% predicted region (pos). If the object is not occluded, then posv is the
% same as pos (posv=[0 0 0 0] by default and indicates posv not set).
% Swapping is used to swap pos and posv, and validating is used to ensure
% pos contains posv. In two cases no swapping/validating occurs. If occl==0
% for a given bb, posv is set to [0 0 0 0]. If occl==1 and posv=[0 0 0 0],
% posv is set to pos. In either case there is no need to swap or validate.
%
% The validate flag:
%  validate==-1:  posv = intersect(pos,posv)
%  validate== 0:  no validation
%  validate==+1:  pos = union(pos,posv)
% If posv is shrunk to 0, it is set to a small bb inside pos.
%
% The swap flag:
%  swap==-1:  pos and posv are swapped before validating
%  swap== 0:  no swapping
%  swap==+1:  pos and posv are swapped after validating
%
% The hide flag:
%  hide==0:  set hide attribute to 0 for all objects
%  hide==1:  set hide attribute to 1 iff object is at some point occluded
%
% Suppose a user has labeled pos in a given video using some annotation
% tool. At this point can swap pos and posv and use the same tool in the
% same manner to label posv (which is stored in pos). Afterwards, can swap
% pos/posv again, and ensure they are consistent, and both end up labeled.
% Additionally, in the second labeling phase, we may want to hide any
% object that is never occluded. The command to setup labeling of posv is:
%  A=vbb('swapPosv',A,-1,1,1); % validate (trust pos) THEN swap
% Afterwards, to swap back, would use:
%  A=vbb('swapPosv',A,1,-1,0); % swap THEN validate (trust posv)
% While labeling posv only frames where occl is already set should be
% altered (the occl flag itself shouldn't be altered).
%
% USAGE
%  A = vbb( 'swapPosv', A, validate, swap, hide )
%
% INPUTS
%  A        - annotation structure
%  validate - see above
%  swap     - see above
%  hide     - see above
%
% OUTPUTS
%  A        - updated annotation
%
% EXAMPLE
%
% see also vbb

for f=1:A.nFrame
  for i=1:length(A.objLists{f})
    o=A.objLists{f}(i); p=o.pos; v=o.posv; vt=[];
    % v is trivial - either [0 0 0 0] or same as p, continue
    if(o.occl==0), vt=[0 0 0 0]; elseif(all(v)==0), vt=p; end
    if(~isempty(vt)), A.objLists{f}(i).posv=vt; continue; end
    % optionally swap before validating
    if(swap==-1), t=p; p=v; v=t; end
    % validate
    if( validate==-1 )
      v = bbApply('intersect',v,p);
      if(all(v==0)), v=[p(1:2)+p(3:4)/2 1 1]; end
    elseif( validate==1 )
      p = bbApply('union',v,p);
    end
    % optionally swap after validating
    if(swap==1), t=p; p=v; v=t; end
    % store results
    o.pos=p; o.posv=v; A.objLists{f}(i)=o;
  end
end

if(~hide), A.objHide(:)=0; else
  for id=find( A.objInit )
    occl=vbb('getVal',A,id,'occl',A.objStr(id),A.objEnd(id));
    A.objHide(id) = all(occl==0);
  end
end

end

function [stats,stats1,logDur] = getStats( A )

% get stats of many annotations simultaneously by first merging
if(length(A)>1), A=merge(A); end

% log activity (allows up to .25h of inactivity)
nObj0=numObj(A); if(nObj0==0), stats=struct(); return; end
log = A.log / 1.1574e-005 / 60 / 60;
locs = find( (log(2:end)-log(1:end-1)) > .25 );
logS=log([1 locs+1]); logE=log([locs A.logLen]);
logDur = sum(logE-logS);

% getStats1 on entire annotation
stats = getStats1( A );

% getStats1 separately for each label
lbl0=unique(A.objLbl); nLbl0=length(lbl0);
stats1=repmat(getStats1(subset(A,lbl0(1))),1,nLbl0);
for i0=2:nLbl0, stats1(i0)=getStats1(subset(A,lbl0(i0))); end

  function stats = getStats1( A )
    % unique labels and label counts
    lbl=unique(A.objLbl); nLbl=length(lbl); lblCnts=zeros(1,nLbl);
    for i=1:nLbl, lblCnts(i)=sum(strcmp(A.objLbl,lbl{i})); end
    stats.labels=lbl; stats.labelCnts=lblCnts;
    % get object lengths
    nObj=numObj(A); stats.nObj=nObj;
    len=A.objEnd-A.objStr+1; stats.len=len;
    % get all per frame info in one flat array ordered by object id
    nPerFrm=cellfun(@length,A.objLists); stats.nPerFrm=nPerFrm;
    ols=A.objLists(nPerFrm>0); ols=[ols{:}];
    ids=[ols.id]; [ids,order]=sort(ids); ols=ols(order); stats.ids=ids;
    inds=[0 cumsum(len)]; stats.inds=inds;
    % get all pos/posv and centers, also first/last frame for each obj
    pos=reshape([ols.pos],4,[])'; posv=reshape([ols.posv],4,[])';
    posS=pos(inds(1:end-1)+1,:); posE=pos(inds(2:end),:);
    stats.pos=pos; stats.posv=posv; stats.posS=posS; stats.posE=posE;
    % get object centers and per frame deltas
    cen=bbApply('getCenter',pos); stats.cen=cen;
    del=cen(2:end,:)-cen(1:end-1,:); del(inds(2:end),:)=-1; stats.del=del;
    % get occlusion information
    occl=(sum(posv,2)>0)'; %occl=[ols.occl]; <--slow
    occFrac=1-posv(:,3).*posv(:,4)./pos(:,3)./pos(:,4); occFrac(occl==0)=0;
    occTime=zeros(1,nObj); for i=1:nObj, occTime(i)=mean(occl(ids==i)); end
    stats.occl=occl; stats.occFrac=occFrac'; stats.occTime=occTime;
  end

  function A = subset( A, lbls )
    % find elements to keep
    nObj=numObj(A); keep=false(1,nObj);
    for i=1:length(lbls), keep=keep | strcmp(A.objLbl,lbls{i}); end
    % clean up objLists by dropping elements
    frms=find(cellfun('isempty',A.objLists)==0); ols=A.objLists;
    for f=frms, ols{f}=ols{f}(keep([ols{f}.id])); end, A.objLists=ols;
    % run cleanup to reorder/drop elements
    A.objInit=keep; A=cleanup(A,0);
  end

  function A = merge( AS )
    nFrm=0; nObj=0;
    for i=1:numel(AS)
      Ai=cleanup(AS(i),0);
      for f=1:Ai.nFrame
        for j=1:length(Ai.objLists{f}),
          Ai.objLists{f}(j).id=Ai.objLists{f}(j).id+nObj;
        end
      end
      Ai.objStr=Ai.objStr+nFrm; Ai.objEnd=Ai.objEnd+nFrm;
      nFrm=Ai.nFrame+nFrm; Ai.nFrame=nFrm;
      nObj=nObj+numObj(Ai); AS(i)=Ai;
    end
    A.nFrame   = nFrm;
    A.objLists = [AS.objLists];
    A.maxObj   = sum([AS.maxObj]);
    A.objInit  = [AS.objInit];
    A.objLbl   = [AS.objLbl];
    A.objStr   = [AS.objStr];
    A.objEnd   = [AS.objEnd];
    A.objHide  = [AS.objHide];
    A.altered  = false;
    A.log      = sort([AS.log]);
    A.logLen   = sum([AS.logLen]);
  end
end

function [gt,posv,lbls1] = frameAnn( A, frame, lbls, test )
% Returns the ground truth bbs for a single frame.
%
% Returns bbs for all object with lbl in lbls. The result is an [nx5] array
% where each row is of the form [x y w h ignore]. [x y w h] is the bb and
% ignore is a 0/1 flag that indicates regions to be ignored. For each
% returned object, the ignore flag is set to 0 if test(lbl,pos,posv)=1 for
% the given object. For example, using lbls={'person','people'}, and
% test=@(lbl,bb,bbv) bb(4)>100, returns bbs for all 'person' and 'people'
% in given frame, and for any objects under 100 pixels tall ignore=1.
%
% USAGE
%  [gt,posv,lbls] = vbb( 'frameAnn', A, frame, lbls, [test] )
%
% INPUTS
%  A        - annotation structure
%  frame    - the frame index
%  lbls     - cell array of string labels
%  test     - [] ignore = ~test(lbl,pos,posv)
%
% OUTPUTS
%  gt       - [n x 5] array containg ground truth for frame
%  posv     - [n x 4] bbs of visible regions
%  lbls     - [n x 1] list of object labels
%
% EXAMPLE
%  lbls={'person','people'}; test=@(lbl,bb,bbv) bb(4)>100;
%  [gt,lbls] = vbb( 'frameAnn', A, 200, lbls, test )

if( nargin<4 ), test=[]; end; assert(frame<=A.nFrame);
ng=length(A.objLists{frame}); ignore=0;
gt=zeros(ng,5); posv=zeros(ng,4); lbls1=cell(1,ng); keep=true(1,ng);
for g=1:ng
  o=A.objLists{frame}(g); lbl=A.objLbl{o.id};
  if(~any(strcmp(lbl,lbls))), keep(g)=0; continue; end
  if(~isempty(test)), ignore=~test(lbl,o.pos,o.posv); end
  gt(g,:)=[o.pos ignore]; lbls1{g}=lbl; posv(g,:)=o.posv;
end
gt=gt(keep,:); lbls1=lbls1(keep); posv=posv(keep,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% helper functions

function A = doubleLen( A )
maxObj    = max(1,A.maxObj);
A.objInit = [A.objInit zeros(1,maxObj)];
A.objLbl  = [A.objLbl   cell(1,maxObj)];
A.objStr  = [A.objStr  -ones(1,maxObj)];
A.objEnd  = [A.objEnd  -ones(1,maxObj)];
A.objHide = [A.objHide zeros(1,maxObj)];
A.maxObj  = max(1,A.maxObj * 2);

A = altered( A );
end

function A = altered( A )
A.altered = true;
if( length(A.log)==A.logLen )
  A.log = [A.log zeros(1,A.logLen)];
end
T = now;  sec=1.1574e-005;
if( A.logLen>0 && (T-A.log(A.logLen))/sec<1 )
  A.log(A.logLen) = T;
else
  A.logLen = A.logLen+1;
  A.log(A.logLen) = T;
end
end

function A = cleanup( A, minn )
% cleanup() Removes placeholder entries from A
if( A.maxObj==0 ), return; end
if( nargin<2 || isempty(minn) ), minn=1; end
% reorder so all initialized objects are first
while( 1 )
  % find first 0 entry in objInit
  [val,id0] = min(A.objInit);
  if( val==1 || id0==A.maxObj ), break; end
  % find last 1 entry past 0 entry
  [val,id1]=max(fliplr(A.objInit(id0+1:end))); id1=A.maxObj-id1+1;
  if(val==0), break; end
  % swap these two locations
  A = swap( A, id0, id1 );
end
% now discard all uninitialized objects (keep at least minn though)
[val,n] = min(A.objInit); n=max(minn,n-1);
if( val==0 )
  A.maxObj  = n;
  A.objInit = A.objInit(1:n);
  A.objLbl  = A.objLbl(1:n);
  A.objStr  = A.objStr(1:n);
  A.objEnd  = A.objEnd(1:n);
  A.objHide = A.objHide(1:n);
end
% discard useless elements in log
A.log = A.log(1:A.logLen);
end

function A = swap( A, id1, id2 )

A0=A;

if(A0.objInit(id1)), fs=A0.objStr(id1):A0.objEnd(id1); else fs=[]; end
for f=fs, ol=A0.objLists{f}; ol([ol.id]==id1).id=id2; A.objLists{f}=ol; end

if(A0.objInit(id2)), fs=A0.objStr(id2):A0.objEnd(id2); else fs=[]; end
for f=fs, ol=A0.objLists{f}; ol([ol.id]==id2).id=id1; A.objLists{f}=ol; end

A.objInit(id1) = A0.objInit(id2);  A.objInit(id2) = A0.objInit(id1);
A.objLbl(id1)  = A0.objLbl(id2);   A.objLbl(id2)  = A0.objLbl(id1);
A.objStr(id1)  = A0.objStr(id2);   A.objStr(id2)  = A0.objStr(id1);
A.objEnd(id1)  = A0.objEnd(id2);   A.objEnd(id2)  = A0.objEnd(id1);
A.objHide(id1) = A0.objHide(id2);  A.objHide(id2) = A0.objHide(id1);

end
