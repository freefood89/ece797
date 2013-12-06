function drawRect_ren(loc,size)
% Draws a rectangle over current plot
x = [loc(1) loc(1)+size(1)-1 loc(1)+size(1)-1 loc(1) loc(1)];
y = [loc(2) loc(2) loc(2)+size(2)-1 loc(2)+size(2)-1 loc(2)];
plot(x,y,'r');
end