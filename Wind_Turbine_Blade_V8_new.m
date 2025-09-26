%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                        %
%               Blade design - Different cord length & twisted angle                     %
%                                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

%=============================%
% START OF USER-SUPPLIED DATA %
%=============================%


%!!Enter your air foil name.
%Airfoil_name = 'ch10';
%Airfoil_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/fx83w160.dat'
%Airfoil_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/fx83w108.dat'
%Airfoil_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/bw3.dat'
%Airfoil_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/fx77080.dat'
Airfoil_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/fx77w121.dat'
%Airfoil_file_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/FX_83-W-160_FX_83-W-160_Re0.012_M0.01_N9.0_360_M'
%Airfoil_file_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/FX_83-W-108_FX_83-W-108_Re0.012_M0.01_N9.0_360_M'
%Airfoil_file_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/Bergey_BW-3__(smoothed)_Bergey_BW-3__(smoothed)_Re0.012_M0.01_N9.0_360_M'
%Airfoil_file_name = 'C:/Users/ecopu/Documents/24-292 Renewable Energy/FX_77-080_FX_77-080_Re0.012_M0.01_N9.0_360_M'
Airfoil_file_name = 'C:\Users\ecopu\Documents\24-292 Renewable Energy\FX_77-W-121_FX_77-W-121_Re0.012_M0.02_N9.0_360_M'

%!!Enter your chord dimensional length (in inches).
Cho_length = 1.5;


%!!Enter your air foil coordinates here:  All of the X-coordinates go in the X_coords array, 
%  and all of the Y-coords go in the Y_coords array.
%  NOTE:  If you imported the data directly into Matlab and set X_coords and Y_coords outside 
%  of this script (in the command window), you should comment out these two lines.
foilxy=importdata([Airfoil_name '.txt']);
X_coords = foilxy.data(:,1);
Y_coords = foilxy.data(:,2);


%!!Insert 20 spanwise locations (in inches) at which cord/twist will be declared.
%  Dis(1)    should be 0 as this is where your blade connects to the shaft.
%  Dis(2:3)  are intermediate locations to generate a smooth transition between the connecting hub and your blade.
%  Dis(4:20) are the locations at which you are controlling the blade shape and where your the chord values you specify in the next section take effect.
%  Example:  Dis = [ 0 : 9.25/19 : 9.25 ];
Dis = [ 0 : 7.4/19 : 7.4 ];%DO NOT EDIT

%!!Enter 20 chord values.  These values are scaling parameters to control the taper of your blade.
%!!The first 3 value are arbitrary because they will be reassigned by code
%  Chord values:  Cho = 0.5  -->  Blade tapers to half of its original size
%                 Cho = 1.0  -->  Blade remains at the original size
%                 Cho = 1.5  -->  Blade flares to 50% bigger than its original size
%  Only use positive real numbers and make sure not to make your blade too big or too small.
%  Example:  Chord = [ 0,0,0, 1 : -0.5/16 : 0.5 ]
%            The above example linearly tapers the blade until it reaches half of its original size
%Examples:
%Cho = [0,0,0,1:-.8/16:.2];%Linear taper from 1 to .2 at tip
%Cho = [0,0,0,.4,.4,.45,.7,1 ,1.05,1.07,1.09,1.07,1.05,1,.9,.8,.65,.5,.35,.1];
%Cho = [0,0,0,1,0.5,0.38,0.29,0.24,0.2,0.17,0.15,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.06]; %BW3_2 & BW3_3
%Cho = [3.274275933, 2.53663687, 1.710474871, 1.238020658, 0.958990481, 0.779320097, 0.655128226, 0.564544524, 0.495706395, 0.441691546, 0.398211394, 0.3624759, 0.332595256, 0.307245578, 0.285472652, 0.26657176, 0.250011379, 0.2500, 0.2500, 0.2500] %BW_4
%Cho = [2.058,1.594,1.075,0.778,0.602,0.490,0.412,0.355,0.312,0.278,0.250,0.228,0.209,0.193,0.179,0.168,0.157,0.148,0.140,0.132]; %FX 83W108 _2
Cho = [0.6875707679, 0.5326825186, 0.3591855252, 0.259974062, 0.2013800411,0.1636507517, 0.1375714897, 0.118549664, 0.1040942283, 0.09275155851,0.08362108747, 0.07611693038, 0.06984224306, 0.06451902101, 0.05994688729,0.05597785693, 0.05250031446, 0.04942851669, 0.04669552414, 0.04424832489];
%Cho(1:20)=1; %constant chord distribution


%!!Enter 20 twist angles (in degrees) to be applied at each of the above spanwise locations.
%  NOTE 1:     Twi_deg(1) should be 0.
%  NOTE 2:     Twist angles can be positive or negative depending on which way you want the blade to twist.
%  NOTE 3:     Make sure you twist your blade in the correct direction.  This depends on the chamber of your blade.
%              If you twist it the wrong direction, the turbine will behave like a propeller instead (which is not what you want).
%  NOTE 4:     The twist degree is an absolute reference!  (This is explained in the example below.)
%  Example 1:  Twi_deg = [0,0,0,0,2,0,0,0,0,0,5,5,5,5,5,5,5,5,5,5];
%              The above example will have no twist for the first 4 locations, then it will twist to 2 degrees,
%              then it will twist BACK to 0 degrees (the original orientation) and stay there for the next 5 points,
%              and finally it will twist to 5 degrees (where it will stay for the rest of the span).
%  Example 2:  The example below will twist linearly from start to finish and end at a twist of 20 degrees.
%Examples:
%Twi_deg = [ 0 : 45/19 : 45 ]; %linear twist
%Twi_deg = 180/pi*atan(60/2.68*(Dis*.0254+0.029)); %inverse tangent twist (constant angle of attack)
%Twi_deg = [0,5,9,13,16,18,20,21,21,21,21,21,21,21,21,21,21,21,21,21];
%Twi_deg = [0, -0.40891675,-0.43573505,-0.43603359,-0.43613315,-0.43618294,-0.43621281,-0.43623273,-0.43624696,-0.43625763,-0.43626592,-0.43627256,-0.43627799,-0.43628252,-0.43628635,-0.43628963,-0.43629248,-0.43629497,-0.43629717,-0.43629912];
%Twi_deg(1:20)=0; %no twist
%Twi_deg = [0,5,9,13,16,18,20,22,24,24,24,24,24,24,24,24,24,24,24,24]; %FX83-W-108
%Twi_deg = [0,5,9,13,16,18,20,22,26,28,30,35,38,38,38,38,38,38,38,38]; %BW3
%Twi_deg = [0,0,5,10,27,29,30,30,30,30,30,30,30,30,30,30,29,27,24,20];
%Twi_deg =[0,47,28,19,13,10,8,6,5,4,3,2.5,2,1.75,1.4,1,0.75,0.5,0.3,0.09];%BW3_2
%Twi_deg = [0, 30.48, 21.0, 15.66, 12.29, 9.98, 8.30, 7.04, 6.05, 5.25, 4.60, 3.59, 2.85, 2.28, 1.83, 1.47, 1.31, 1.17, 1.04, 0.92]; %BW3_3
%Twi_deg = [0, 47.18066761, 27.89323576, 18.61636181, 13.43083236, 10.1692931, 7.941835455, 6.328443933, 5.107731294, 4.152691243, 3.385500044, 2.75590772, 2.230061025, 1.784339163, 1.401768872, 1.069844653, 0.779152276, 0.522470791, 0.294169909, 0.089795452] %BW_4
%Twi_deg = [0,25.78,6.49,-2.78,-7.97,-11.23,-13.46,-15.07,-16.29,-17.25,-18.01,-18.64,-19.17,-19.61,-20,-20.33,-20.88,-20.62,-20.88,-21.11];
%Twi_deg = 
Twi_deg = [0,0,7.5,15,40.5,44,45,45,45,45,45,45,45,45,45,45,43.5,40.5,36,30];


%Input CL, CD, vs Alpha from XFOIL analysis

coefdata = importdata([Airfoil_file_name '.dat']);
%coefdata = load(['test.pol']);
alfa_de = coefdata.data(:,1);
alfa_deg = alfa_de(1:end-1)
cll = coefdata.data(:,2);
cl = cll(1:end-1)
cdd = coefdata.data(:,3);
cd = cdd(1:end-1)
disp(alfa_deg)
disp(cl)
disp(cd)


%===========================%
% END OF USER-SUPPLIED DATA %
%===========================%


%% Normalize and calculate blade points


nvals = 20;

%{
%error check
if( length(X_coords) < 3 || length(Y_coords) < 3 )
    fprintf('\n\nError:  X_coords/Y_coords arrays do not contain valid data!\n\n')
    return
end
if( length(X_coords) ~= length(Y_coords) )
    fprintf('\n\nError:  Lengths of X_coords and Y_coords do not match!\n\n')
    return
end
if( length(Dis) ~= nvals )
    fprintf('\n\nError:  Dis should have 20 entries!\n\n')
    return
end
if( length(Twi_deg) ~= nvals )
    fprintf('\n\nError:  Twi_deg should have 20 entries!\n\n')
    return
end
if( length(Cho) ~= nvals )
    fprintf('\n\nError:  Chord should have 20 entries!\n\n')
    return
end
if( length(alfa_deg) ~= length(cl) )
    fprintf('\n\nError:  Lengths of alpha and Coef_lift do not match!\n\n')
    return
end
if( length(alfa_deg) ~= length(cd) )
    fprintf('\n\nError:  Lengths of alpha and Coef_drag do not match!\n\n')
    return
end
%}

%hub diameter (DO NOT CHANGE!)
HD = 0.25;

%values are normalized based on Span Length
norm = Dis(nvals);
Dis_No = Dis ./ norm;
HD_No = HD ./ norm;
%      The first two are connector piece.  Second two are determined below in order to be make smooth transition
Cho(1) = HD;
Cho_No = Cho ./ norm;

m = (Cho_No(4)-Cho_No(1))/(Dis_No(4)-Dis_No(1));

for i = 2 : 3
    %This assigns chords values that were Zero(arbitrary) above
    Cho_No(i) = m*(Dis_No(i)-Dis_No(1)) + Cho_No(1);
end    

%Converts degree to radian. Also switches sign, for twist to be more intuitive
Twi_rad = zeros(1,20);
for i = 1 : 20
    Twi_rad(i) = -Twi_deg(i)*(pi/180);
end

npts = ((length(X_coords)+1)/2);%**May need to edit this, depending if you have even or odd number of panels

%!!Insert upper and lower coordinate of profile.  Unlike XFoil, the LE and TE are repeated for the upper and lower sections here.
% coordinates_upper
Xup_co = X_coords(1:npts);
Yup_co = Y_coords(1:npts);
% coordinates_lower
Xlo_co = X_coords(npts:length(X_coords));
Ylo_co = Y_coords(npts:length(Y_coords));

npts=floor(npts);

Xup_tw     = zeros(npts,20);
Yup_tw     = zeros(npts,20);
Xlo_tw     = zeros(npts,20);
Ylo_tw     = zeros(npts,20);


%Scale x values for round shaft cross-section
Xup_tw(:,1) = Xup_co .* Cho_No(1) - Cho_No(1)/2;
Xlo_tw(:,1) = Xlo_co .* Cho_No(1) - Cho_No(1)/2;
%Calculate y values from circle formula
for k = 1 : npts
    Yup_tw(k,1) = abs(sqrt(((Cho_No(1)/2)^2) - Xup_tw(k)^2));
end
for k = 1 : npts
    Ylo_tw(k,1) = abs(sqrt(((Cho_No(1)/2)^2) - Xlo_tw(k)^2)) .* (-1);
end


%Plots the round cross-section which matches shaft 
figure (1)
clf
R = ones(1,npts).*Dis_No(1);
plot3(R',norm*Xup_tw(:,1),norm*Yup_tw(:,1),'r',R',norm*Xlo_tw(:,1),norm*Ylo_tw(:,1),'b');
hold('on');

Xup_co_cho = zeros(npts,nvals);
Xup        = zeros(npts,nvals);
Yup_co_cho = zeros(npts,nvals);
Yup        = zeros(npts,nvals);
Xlo_co_cho = zeros(npts,nvals);
Xlo        = zeros(npts,nvals);
Ylo_co_cho = zeros(npts,nvals);
Ylo        = zeros(npts,nvals);
for i = 2 : nvals
    Rate_X = Cho_No(i) * 0.3;
    for k=1:npts
        Xup_co_cho(k,i) = Xup_co(k)* Cho_No(i);
        Xup(k,i) = Xup_co_cho(k,i) - Rate_X;     
        Yup_co_cho(k,i) = Yup_co(k)* Cho_No(i);
        Yup(k,i) = Yup_co_cho(k,i);
        Xup_tw(k,i) = cos(Twi_rad(i)) * (Xup(k,i)) - sin(Twi_rad(i)) * (Yup(k,i));      
        Yup_tw(k,i) = sin(Twi_rad(i)) * (Xup(k,i)) + cos(Twi_rad(i)) * (Yup(k,i));
    end
    for k = 1:npts
        Xlo_co_cho(k,i) = Xlo_co(k)* Cho_No(i);
        Xlo(k,i) = Xlo_co_cho(k,i) - Rate_X;
        Ylo_co_cho(k,i) = Ylo_co(k)* Cho_No(i);
        Ylo(k,i) = Ylo_co_cho(k,i);
        Xlo_tw(k,i) = cos(Twi_rad(i)) * (Xlo(k,i)) - sin(Twi_rad(i)) * (Ylo(k,i));
        Ylo_tw(k,i) = sin(Twi_rad(i)) * (Xlo(k,i)) + cos(Twi_rad(i)) * (Ylo(k,i));
    end
    R = ones(1,npts);
    E = ones(1,npts);
    R = Dis(nvals)*R .*Dis_No(i);
    E = Dis(nvals)*E .*Dis_No(i);
    plot3(R,norm*Xup_tw(:,i),norm*Yup_tw(:,i),'r',E,norm*Xlo_tw(:,i),norm*Ylo_tw(:,i),'b');
    
end

%Plots orthogonally to others plots, producing hashed surface
for k = 1 : npts
    plot3(Dis(nvals)*Dis_No(1:nvals),norm*Xup_tw(k,1:nvals),norm*Yup_tw(k,1:nvals),'r');
end
for k = 1 : npts
    plot3(Dis(nvals)*Dis_No(1:nvals),norm*Xlo_tw(k,1:nvals),norm*Ylo_tw(k,1:nvals),'b');
end



grid on
% xlim([-1,1]);
% ylim([-1,1]);
% zlim([-1,1]);
% xlim([-1,Dis_No(20)]);
% ylim([Cho_length*min(min(Xlo_tw(k,1:20))),Cho_length*max(max(Xlo_tw(k,1:20)))]);
% zlim([Cho_length*min(min(Ylo_tw(k,1:20))),Cho_length*max(max(Ylo_tw(k,1:20)))]);

% axis auto
axis([0,75/8,-5,5,-3,3]);
% xlim([0,75/8]);
% ylim([-2.25/2,2.25/2]);
% zlim([-0.75/2,0.75/2]);
title(Airfoil_name)

%% Output data to file NEW VERSION


% fprintf(fid,'     X             Y             Z      \n');
% fprintf(fid,'============  ============  ============');

Xup_tw_ln = norm*Xup_tw;
Yup_tw_ln = norm*Yup_tw;
Xlo_tw_ln = norm*Xlo_tw;
Ylo_tw_ln = norm*Ylo_tw;

%Fatten up connecting neck
% Ylo_tw_ln(:,2:3) = Ylo_tw_ln(:,2:3)*1.5;
% Yup_tw_ln(:,2:3) = Yup_tw_ln(:,2:3)*1.5;

for i = 1:nvals
    fid = fopen(['C:/Users/ecopu/Documents/24-292 Renewable Energy/SLDCRV_fx77w121/blade' int2str(i) '.sldcrv'],'w');
    % Special handling of first line in the file
    k=1;
    fprintf(fid,'%12.8f  %12.8f  %12.8f',Xup_tw_ln(k,i),Yup_tw_ln(k,i),Dis(i));
    
    
    % Write out the points on the upper surface
    for k = 2:npts
        fprintf(fid,'\n%12.8f  %12.8f  %12.8f',Xup_tw_ln(k,i),Yup_tw_ln(k,i),Dis(i));
    end
    
    % Write out the points on the lower surface
    if Yup_tw_ln(npts) == Ylo_tw_ln(1)
        kstart=2;
    else
        kstart=1;
    end
    for k = kstart:npts
        fprintf(fid,'\n%12.8f  %12.8f  %12.8f',Xlo_tw_ln(k,i),Ylo_tw_ln(k,i),Dis(i));
    end
    
    % Write out the first point of upper, to close surface
    % Check file to make sure line isn't ALREADY repeated, if so, comment
    % out these two lines
    if Yup_tw_ln(1,i) ~= Ylo_tw_ln(npts,i)
        fprintf(fid,'\n%12.8f  %12.8f  %12.8f',Xup_tw_ln(1,i),Yup_tw_ln(1,i),Dis(i));
    end
    
    fclose(fid);
end

%% Estimate performance

%genspecs=load('rf500t.dat');

alfa_rad = alfa_deg*pi/180;

%END of user supplied data
%Quantities use SI units 
%NOTE: airfoil coordinate values are in inches, chord/distance values are
%converted here to meters

v=4;%incoming wind velocity (m/s)
b=.029;%hub radius (m)
rho=1.18;%air density (kg/m^3)
Cho_si=Cho_No*Dis(nvals)*.0254;%returns chord distribution in meters
Dis_si=Dis*0.0254+b;%returns distances in meters (measured from hub center)
dx=Dis_si(2)-Dis_si(1);

a0max=0;
a0pmax=0;
torque=zeros(20,1);
torquemax=zeros(20,1);
torquepmax=zeros(20,1);
lift=zeros(20,1);
liftmax=zeros(20,1);
liftpmax=zeros(20,1);
drag=zeros(20,1);
dragmax=zeros(20,1);
dragpmax=zeros(20,1);
alphaeff=zeros(20,1);
alphaeffmax=zeros(20,1);
alphaeffpmax=zeros(20,1);
phi_deg=zeros(20,1);
phi_degmax=zeros(20,1);
phi_degpmax=zeros(20,1);
wmax=0;
wpmax=0;
pmax=0;
for k=1:1:180%this loop steps through base angle at hub (adjustable by hand)
    a0=k*pi/180/2;
    wacc=0;%for accumulation to view power curve for blade
    tacc=0;
    for j=1:1:1000%this loop steps through spin speeds
        w=j*.5;
        T=0;
        for i=3:1:20%adds up torque over length of blade
            veff=sqrt(v^2+(w*Dis_si(i))^2);
            phi=atan(w*Dis_si(i)/v);
            phi_deg(i)=phi*180/pi;
            alpha=-Twi_rad(i)+a0-phi;
            alphaeff(i)=alpha;
            if alpha>max(alfa_rad)
                L=0.3*max(cl)*0.5*rho*veff^2*Cho_si(i)*dx;
                D=max(cd)*0.5*rho*veff^2*Cho_si(i)*dx;
            elseif alpha<min(alfa_rad)
                L=min(cl)*0.5*rho*veff^2*Cho_si(i)*dx;
                D=max(cd)*0.5*rho*veff^2*Cho_si(i)*dx;
            else
                L=interp1(alfa_rad,cl,alpha)*0.5*rho*veff^2*Cho_si(i)*dx;
                D=interp1(alfa_rad,cd,alpha)*0.5*rho*veff^2*Cho_si(i)*dx;
            end
            T=T+Dis_si(i)*(cos(phi)*L-sin(phi)*D);
            torque(i)=Dis_si(i)*(cos(phi)*L-sin(phi)*D);
            lift(i)=cos(phi)*L;
            drag(i)=sin(phi)*D;
        end
%         if T<0
%             break
%         end
        wacc(j)=w;
        tacc(j)=3*T;
        if w>wmax
            wmax=w;
            a0max=a0;
            torquemax=torque;
            liftmax=lift;
            dragmax=drag;
            alphaeffmax=alphaeff;
            phi_degmax=phi_deg;
            wpow=wacc;%%!!double check that using same angle
            tpow=tacc;
        end
        if w*T>pmax
            pmax=w*3*T;
            wpmax=w;
            a0pmax=a0;
            torquepmax=torque;
            liftpmax=lift;
            dragpmax=drag;
            alphaeffpmax=alphaeff;
            phi_degpmax=phi_deg;
        end
        
        if T<0
            if a0 == a0pmax
                wpow=wacc;
                tpow=tacc;
            end
            break
        elseif w==500
            disp('torque never went negative')
        end
    end
%     if a0-Twi_rad(nvals)>pi/4*1.1
%         break
%     end
end

%Output data for Maximum spin rate
for i=1:1:2
    phi=atan(wmax*Dis_si(i)/v);
    phi_degmax(i)=phi*180/pi;
    alfa_deg=-Twi_rad(i)+a0max-phi;
    alphaeffmax(i)=alfa_deg;
end

disp('RESULTS:')
disp(['Maximum spin rate of ' num2str(wmax) 'rad/s occurs at a mounting angle of ' num2str(a0max) 'rad.'])
alphaeff_deg=alphaeffmax*180/pi;

figure(2)
clf
subplot(2,2,[1 3])
hold on
plot(Dis_si/.0254,Twi_deg+a0max*180/pi,'b')
plot(Dis_si/.0254,phi_degmax,'r')
plot(Dis_si/.0254,alphaeff_deg,'g')
plot(Dis_si/.0254,zeros(1,nvals),'--k')
title('Distribution of angles')
xlabel('Distance from hub center')
ylabel('degrees')
legend('Blade mounting angle','Apparent direction of wind','Effective angle of attack','zero-point')
legend('Location','East')
legend('boxoff')

subplot(2,2,2)
hold on
plot(Dis_si/.0254,liftmax-dragmax,'k')
plot(Dis_si/.0254,liftmax,'g')
plot(Dis_si/.0254,-dragmax,'r')
plot(Dis_si/.0254,zeros(1,nvals),'--k')
title('Force Distributions')
xlabel('Distance from hub')
ylabel('N')
legend('Total force','Lift Force','Drag Force','zero-point')
legend('Location','NorthEast')
legend('boxoff')

subplot(2,2,4)
hold on
plot(Dis_si/.0254,torquemax)
plot(Dis_si/.0254,zeros(1,nvals),'--k')
title('Torque Distribution')
xlabel('Distance from hub')
ylabel('N*m')

% %Output data for Maximum power output
% for i=1:1:2
%     phi=atan(wpmax*Dis_si(i)/v);
%     phi_degpmax(i)=phi*180/pi;
%     alfa_deg=-Twi_rad(i)+a0pmax-phi;
%     alphaeffpmax(i)=alfa_deg;
% end
% %Calculate resisitor for peak power electrical load
% if wpmax>genspecs(3,1)
%     vpmax=(genspecs(3,2)-genspecs(2,2))/(genspecs(3,1)-genspecs(2,1))*(wpmax-genspecs(3,1))+genspecs(3,2);
% elseif wpmax>genspecs(3,1)
%     vpmax=(genspecs(1,2)-genspecs(2,2))/(genspecs(1,1)-genspecs(2,1))*(wpmax-genspecs(1,1))+genspecs(1,2);
% else
%     vpmax=interp1(genspecs(:,1),genspecs(:,2),wpmax);
% end
% r=vpmax^2/pmax;
% 
% %Calculate turbine vs circuit power curves
% n=length(wpow);
% vpow=zeros(1,n);
% for i=1:n
%     if wpow(i)>genspecs(3,1)
%         vpow(i)=(genspecs(3,2)-genspecs(2,2))/(genspecs(3,1)-genspecs(2,1))*(wpow(i)-genspecs(3,1))+genspecs(3,2);
%     elseif wpow(i)<genspecs(1,1)
%         vpow(i)=(genspecs(1,2)-genspecs(2,2))/(genspecs(1,1)-genspecs(2,1))*(wpow(i)-genspecs(1,1))+genspecs(1,2);
%     else
%         vpow(i)=interp1(genspecs(:,1),genspecs(:,2),wpow(i));
%     end
% end

% disp(['Maximum power output of ' num2str(pmax) 'W occurs at a spin rate of ' num2str(wpmax) 'rad/s, with a mounting angle of ' num2str(a0pmax) 'rad.'])
% disp(['Suggested to use ' num2str(r) 'ohm resistor.'])
% alphaeff_deg=alphaeffpmax*180/pi;

% figure(3)
% clf
% subplot(2,2,[1 3])
% hold on
% plot(Dis_si/.0254,Twi_deg+a0pmax*180/pi,'b')
% plot(Dis_si/.0254,phi_degpmax,'r')
% plot(Dis_si/.0254,alphaeff_deg,'g')
% plot(Dis_si/.0254,zeros(1,nvals),'--k')
% title('Distribution of angles')
% xlabel('Distance from hub center')
% ylabel('degrees')
% legend('Blade mounting angle','Apparent direction of wind','Effective angle of attack','zero-point')
% legend('Location','East')
% legend('boxoff')
% 
% subplot(2,2,2)
% hold on
% plot(Dis_si/.0254,liftpmax-dragpmax,'k')
% plot(Dis_si/.0254,liftpmax,'g')
% plot(Dis_si/.0254,-dragpmax,'r')
% plot(Dis_si/.0254,zeros(1,nvals),'--k')
% title('Force Distributions')
% xlabel('Distance from hub')
% ylabel('N')
% legend('Total force','Lift Force','Drag Force','zero-point')
% legend('Location','NorthEast')
% legend('boxoff')
% 
% subplot(2,2,4)
% hold on
% plot(Dis_si/.0254,torquepmax)
% plot(Dis_si/.0254,zeros(1,nvals),'--k')
% title('Torque Distribution')
% xlabel('Distance from hub')
% ylabel('N*m')
% 
% figure(4)
% clf
% hold on
% plot(wpow,tpow.*wpow,'b')
% plot(wpow,vpow.^2/r,'r')
% title('Power Curves')
% xlabel('spin speed (rad/s)')
% ylabel('power (Watt)')
% legend('Power output by turbine blades','Power dissipated ')
% legend('Location','NorthWest')
% legend('boxoff')
% % figure(2)
% % hold off
% % plot(alfa,cl)
% % hold on
% % plot(alfa,cd)
% % figure(3)
% % plot(cl,cd)