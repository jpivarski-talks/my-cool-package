
//constants - button margin, width, height and spacing
int BL=720;
int BT=0;
int BW=70;
int BH=24;
int BS=28;
int MY=182;
int MX=150;

boolean showGRID=true;
boolean showLABELS=true;
boolean showVALUES=false;
boolean showCOS=true;
boolean showSIN=false;
boolean showVERSIN=false;
boolean showSEC=false;
boolean showTAN=false;
boolean showCOT=false;
boolean showEXSEC=false;
boolean showCSC=false;
boolean showVERCOS=false;
boolean showEXCSC=false;


//variables
int R=100;
float angle=0;
String lab;


void setup() {
  size(800, 364, P2D);

  }

void draw() {

//Background
background(0,0,0,0);

//angle
angle=atan2(MY-float(mouseY), float(mouseX)-MX);
textSize(20);
textAlign(LEFT, TOP);
text("Angle: "+nf(angle,1,3),10,0);



//circle
noFill();
stroke(255,255,255);
strokeWeight(2);
ellipse(MX,MY,R*2,R*2);

//gridlines
if (showGRID) {
  strokeWeight(1);
  stroke(100,100,100);
  line(10,MY,BL-10,MY);
  line(MX,10,MX,350);
  line(MX+(R*cos(angle)),MY,MX+(R*cos(angle)),MY-(R*sin(angle))+(10*sgn(mouseY-MY)));
  line(MX,MY-(R*sin(angle)),MX+(R*cos(angle))+(10*sgn(mouseX-MX)),MY-(R*sin(angle)));
}

//cos
if (showCOS) {
  strokeWeight(1);
  stroke(255,0,0);
  line(MX,MY-(R*sin(angle)),MX+(R*cos(angle)),MY-(R*sin(angle)));
  fill(255,0,0);rect(BL+BW+3,leg(3),7,7);
  if (showLABELS) {lab="Cos";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(cos(angle),1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(CENTER, BOTTOM);
                   fill(255,0,0);
                   text(lab,MX+((R/2)*cos(angle)), MY-(R*sin(angle)));
                  }
}

//sin
if (showSIN) {
  strokeWeight(1);
  stroke(0,255,0);
  line(MX+(R*cos(angle)),MY,MX+(R*cos(angle)),MY-(R*sin(angle)));
  fill(0,255,0);rect(BL+BW+3,leg(4),7,7);
  if (showLABELS) {lab="Sin";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(sin(angle),1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(LEFT, CENTER);
                   fill(0,255,0);
                   text(lab,MX+((R)*cos(angle))+5, MY-((R/2)*sin(angle)));
                  }
}


//versin
if (showVERSIN) {
  strokeWeight(1);
  stroke(128,64,255);
  line(MX+(R*cos(angle)),MY,MX+R,MY);
  fill(128,64,255);rect(BL+BW+3,leg(9),7,7);
  if (showLABELS) {lab="Versin";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(1-cos(angle),1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(CENTER, TOP);
                   fill(128,64,255);
                   text(lab,MX+R-(R*(1-cos(angle))/2), MY+2);
                  }
}

//sec
if (showSEC) {
  stroke(255,255,0);
  fill(255,255,0);rect(BL+BW+3,leg(7),7,7);
  float v;
  if (cos(angle)!=0) {
  v=1/cos(angle);
  strokeWeight(1);
  line(MX,MY,MX+(R*v),MY);

  if (showLABELS) {lab="Sec";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(v,1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(CENTER, TOP);
                   fill(255,255,0);
                   text(lab,MX+(v*R*.5), MY+2);
                  }
  }
}

//excsc
if (showEXCSC) {
  stroke(0,148,255);
  fill(0,148,255);rect(BL+BW+3,leg(12),7,7);
  float v;
  if (sin(angle)!=0) {
  v=1/sin(angle);
  v=v-1;
  strokeWeight(1);
  line(MX,MY-R,MX,MY-R-(R*v));

  if (showLABELS) {lab="Excsc";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(v,1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(LEFT, CENTER);
                   fill(0,148,255);
                   text(lab,MX+2, ((MY-R)+(MY-R-(R*v)))/2);
                  }
  }
}


//CSC
if (showCSC) {
  stroke(255,180,128);
  fill(255,180,128);rect(BL+BW+3,leg(6),7,7);
  float v;
  if (sin(angle)!=0) {
  v=1/sin(angle);
  strokeWeight(1);
  line(MX,MY-(R*v),MX,MY);

  if (showLABELS) {lab="Csc";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(v,1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(LEFT, CENTER);
                   fill(255,180,128);
                   text(lab,5+MX, MY-(v*R)/2);
                  }
  }
}



//exsec
if (showEXSEC) {
  stroke(64,255,164);
  fill(64,255,164);rect(BL+BW+3,leg(11),7,7);
  float v;
  if (cos(angle)!=0) {
  v=1/cos(angle);
  v=v-1;
  strokeWeight(1);
  line(MX+R,MY,MX+R+(R*v),MY);

  if (showLABELS) {lab="Exsec";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(v,1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(CENTER, TOP);
                   fill(64,255,164);
                   text(lab,MX+R+(R*(v/2)), MY+2);
                  }
  }
}

//vercos
if (showVERCOS) {
  stroke(180,0,255);
  fill(180,0,255);rect(BL+BW+3,leg(10),7,7);
  strokeWeight(1);
  line(MX,MY-(R*sin(angle)),MX,MY-R);

  if (showLABELS) {lab="Vercos";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(1-sin(angle),1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(LEFT, CENTER);
                   fill(180,0,255);
                   text(lab,MX+5, ((MY-(R*sin(angle))+(MY-R))/2));
                  }
}


//tan
if (showTAN) {
  stroke(255,128,0);
  fill(255,128,0);rect(BL+BW+3,leg(5),7,7);
  float v;
  if (cos(angle)!=0) {
  v=1/cos(angle);
  strokeWeight(1);
  line(MX+(R*cos(angle)),MY-(R*sin(angle)),MX+(R*v),MY);

  if (showLABELS) {lab="Tan";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(tan(angle),1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(LEFT, CENTER);
                   fill(255,128,0);
                   text(lab,abs(tan(angle)*10)+(R*cos(angle))+MX+((R*v)-(R*cos(angle)))/2, MY-((R/2)*sin(angle)));
                  }
  }
}

//cot
if (showCOT) {
  stroke(128,255,255);
  fill(128,255,255);rect(BL+BW+3,leg(8),7,7);
  float v;
  if (sin(angle)!=0) {
  v=1/sin(angle);
  strokeWeight(1);
  line(MX,MY-(R*v),MX+(R*cos(angle)),MY-(R*sin(angle)));

  if (showLABELS) {lab="Cot";} else {lab="";}
  if (showVALUES) {lab=lab+" "+nf(1/tan(angle),1,3);}
  if (showLABELS||showVALUES)
                  {
                   textSize(14);
                   textAlign(LEFT, CENTER);
                   fill(128,255,255);
                   text(lab,10+MX+(R*cos(angle))/2, ((MY-(R*v))+(MY-(R*sin(angle))))/2 );
                  }
  }
}



//line trace
if (showGRID) {
stroke(100,100,100);
strokeWeight(1);
line(MX,MY,mouseX,mouseY);
}

stroke(255,255,255);
line(MX,MY,MX+(R*cos(angle)),MY-(R*sin(angle)));
//bulb
fill(255,255,255);
ellipse(MX+(R*cos(angle)),MY-(R*sin(angle)),7,7);

//buttons

dbutton(0,"Lines",showGRID);
dbutton(1,"Labels",showLABELS);
dbutton(2,"Values",showVALUES);

dbutton(3,"Cos",showCOS);
dbutton(4,"Sin",showSIN);
dbutton(5,"Tan",showTAN);
dbutton(6,"Csc",showCSC);
dbutton(7,"Sec",showSEC);
dbutton(8,"Cot",showCOT);

dbutton(9,"Versin",showVERSIN);
dbutton(10,"Vercos",showVERCOS);
dbutton(11,"Exsec",showEXSEC);
dbutton(12,"Excsc",showEXCSC);


}

int leg(int p)
{
  return BT+(BS*p)+(BS/2)-4;
}

void dbutton(int a, String s, boolean c)
{
 String nl;
 strokeWeight(1);
 stroke(255,255,255);
 textAlign(CENTER, CENTER);
 textSize(12);
 if (c)  {fill(64,164,64);} else {fill(164,64,64);}
 rect(BL,BT+(a*BS),BW,BH,5);
 fill(255,255,255);
 if (!c) {nl=s+" [X]";} else {nl=s;}
 text(nl,BL+(BW/2),BT+(BH/2)+(a*BS));
}


int sgn(int v)
{
  if (v>=0) {return 1;} else {return -1;}
}

// functions

void mousePressed() {

if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=BT && mouseY<=(BT+BH) ) {showGRID=!showGRID;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*1)) && mouseY<=(BT+(BS*1)+BH)) {showLABELS=!showLABELS;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*2)) && mouseY<=(BT+(BS*2)+BH)) {showVALUES=!showVALUES;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*3)) && mouseY<=(BT+(BS*3)+BH)) {showCOS=!showCOS;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*4)) && mouseY<=(BT+(BS*4)+BH)) {showSIN=!showSIN;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*5)) && mouseY<=(BT+(BS*5)+BH)) {showTAN=!showTAN;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*6)) && mouseY<=(BT+(BS*6)+BH)) {showCSC=!showCSC;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*7)) && mouseY<=(BT+(BS*7)+BH)) {showSEC=!showSEC;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*8)) && mouseY<=(BT+(BS*8)+BH)) {showCOT=!showCOT;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*9)) && mouseY<=(BT+(BS*9)+BH)) {showVERSIN=!showVERSIN;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*10)) && mouseY<=(BT+(BS*10)+BH)) {showVERCOS=!showVERCOS;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*11)) && mouseY<=(BT+(BS*11)+BH)) {showEXSEC=!showEXSEC;}
if (mouseButton==LEFT && mouseX>=BL && mouseX<=(BL+BW) && mouseY>=(BT+(BS*12)) && mouseY<=(BT+(BS*12)+BH)) {showEXCSC=!showEXCSC;}

}
