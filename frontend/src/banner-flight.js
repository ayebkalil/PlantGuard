export const bannerCredit = 'made by yasmine , ibrahime and othmane';
// The upper panel holds the banner and airframe. The strip below it holds
// sixteen views of the nose over half a turn of the two-bladed propeller.
export const flightSheets = [
  {url:'/botany/banner-flight.webp',width:1280,height:192,nose:[1227,17,40,138]},
  {url:'/botany/banner-flight-compact.webp',width:960,height:200,nose:[922,39,30,104]}
];
const frameCount=16, gutter=4;

export function flightPose(width, height, time, still) {
  const compact = width < 620;
  const span = compact ? Math.min(width * .93, 420) : Math.min(540, Math.max(415, width * .39));
  const sheet = flightSheets[Number(compact)];
  const tall = span * sheet.height / sheet.width;
  const distance = width + span;
  const duration = compact ? 49 : 44;
  const phase = still ? .5 : ((time / duration + .5 - 12 / duration) % 1);
  const x = phase * distance - span;
  const altitude = Math.max(height * (compact ? .145 : .12), Math.min(86, height*.19));
  const y = altitude - tall / 2 + (still ? 0 : Math.sin(phase * Math.PI * 2) * tall * .09);
  const frame=still ? 0 : Math.floor(time*64)%frameCount;
  const sourceX=gutter+frame*(sheet.nose[2]+2*gutter);
  const sourceY=sheet.height+gutter;
  return { x, y, span, tall, sheet, compact, sourceX, sourceY, sheetHeight:sourceY+sheet.nose[3]+gutter };
}
