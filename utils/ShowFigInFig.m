
function [imWhole,imPiece,varargout] = ShowFigInFig(im, PosStart, wSize, lineSize, BoxColor, varargin)
% Show figure in a figure. First, draw a box in the image and 
% also copy this piece of image in the box, at last, 
% show the cropped piece in the image,too.

%Input��
% im:	original image, RGB or gray.
% PosStart�� Starting position for drawing the box, i.e., the coordinate of the top left point.
% wSize��   [wx, wy], the size of the box.
% lineSize�� the width of the box line.
% BoxColor��    [r,  g,  b] , the color of the box
%----------------------------------------------------------------------
%Output��
% imWhole�� the whole image with a box in it.
% imPiece: the piece of image in that box.

% Example
% [imWhole,imPiece] = ShowFigInFig(im, PosStart, wSize, lineSize,
% BoxColor);

% 23/11/2016 Code finished.

imWhole = im;
[m, n, z] = size(imWhole);

RecArea = [PosStart, wSize]; % the area in the box

imPiece = imcrop(imWhole,RecArea); % cropped image piece

PosStart_Piece = [1, 1]; % Starting position for drawing the box in the image piece.

wSize_Piece = size(imPiece(:,:,1))-[1, 1]; % size of the box.


imWhole = drawBox(imWhole,PosStart,wSize,lineSize, BoxColor );

imPiece = drawBox(imPiece,PosStart_Piece,wSize_Piece,1, BoxColor );


%%
show_figure = 0;
save_figure = 0;

if show_figure
	figure;
	imshow(imWhole); hold on;
	set(gcf,'Position', [100,100,ImWidth,ImHeigth]);
	set(gca, 'Position', [0,0,1,1]);

	axes('Position',[0,0,0.4,0.4]);
	imshow(imPiece)
	pause(0.2);
end


if save_figure
	addpath('../export_fig');
	FigFormat = ['.png'] ; % ['.jpg'] ; % ['.eps'];
% 	current_date = date;
	DATE = datestr(now,30); % 30 (ISO 8601)  'yyyymmddTHHMMSS'        20000301T154517 
	FileName = ['Im_' , DATE];
	export_fig(gcf, [FileName, FigFormat])  % export_fig(figure_handle, filename);		
end

end
		

function [ dest ] = drawBox( im, PosStart, wSize,  lineSize, color )
%��飺
% %��ͼ��������ɫ�Ŀ�ͼ����������ǻҶ�ͼ����ת��Ϊ��ɫͼ���ٻ���ͼ
% ͼ�����
% ����������  ��  y
% ����������  ��  x
%----------------------------------------------------------------------
%���룺
% im��        ԭʼͼ�񣬿���Ϊ�Ҷ�ͼ����Ϊ��ɫͼ
% PosStart��         ���Ͻ�����   [x1, y1]
% wSize��   ��Ĵ�С      [wx, wy]
% lineSize�� �ߵĿ��
% color��     �ߵ���ɫ      [r,  g,  b] 
%----------------------------------------------------------------------
%�����
% dest��           �����˵�ͼ��
%----------------------------------------------------------------------

%flag=1: ��ȱ�ڵĿ�
%flag=2: ��ȱ�ڵĿ�
flag = 2;


%�ж������������
if nargin < 5
    color = [255 255 0];
end

if nargin < 4
    lineSize = 1;
end

if nargin < 3
    disp('����������� !!!');
    return;
end





%�жϿ�ı߽�����
[yA, xA, z] = size(im);
x1 = PosStart(1);
y1 = PosStart(2);
wx = wSize(1);
wy = wSize(2);
if  x1>xA || ...
        y1>yA||...
        (x1+wx)>xA||...
        (y1+wy)>yA

    disp('���Ŀ򽫳���ͼ�� !!!');
    return;
end

%����ǵ�ͨ���ĻҶ�ͼ��ת��3ͨ����ͼ��
if 1==z
    dest(:, : ,1) = im;
    dest(:, : ,2) = im;
    dest(:, : ,3) = im;
else
    dest = im;
end

%��ʼ����ͼ
for c = 1 : 3                 %3��ͨ����r��g��b�ֱ�
    for dl = 1 : lineSize   %�ߵĿ�ȣ���������������չ��
        d = dl - 1;
        if  1==flag %��ȱ�ڵĿ�
            dest(  y1-d ,            x1:(x1+wx) ,  c  ) =  color(c); %�Ϸ�����
            dest(  y1+wy+d ,     x1:(x1+wx) , c  ) =  color(c); %�·�����
            dest(  y1:(y1+wy) ,   x1-d ,           c  ) =  color(c); %������
            dest(  y1:(y1+wy) ,   x1+wx+d ,    c  ) =  color(c); %������
        elseif 2==flag %��ȱ�ڵĿ�
            dest(  y1-d ,            (x1-d):(x1+wx+d) ,  c  ) =  color(c); %�Ϸ�����
            dest(  y1+wy+d ,    (x1-d):(x1+wx+d) ,  c  ) =  color(c); %�·�����
            dest(  (y1-d):(y1+wy+d) ,   x1-d ,           c  ) =  color(c); %������
            dest(  (y1-d):(y1+wy+d) ,   x1+wx+d ,    c  ) =  color(c); %������
        end
    end    
end %��ѭ��β


end %����β