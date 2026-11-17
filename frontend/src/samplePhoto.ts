export function createSamplePhotoDataUrl(preset: 'portrait' | 'landscape' | 'cybercar'): {
  dataUrl: string;
  filename: string;
} {
  const canvas = document.createElement('canvas');
  canvas.width = 384;
  canvas.height = 384;
  const ctx = canvas.getContext('2d')!;

  if (preset === 'portrait') {
    const bg = ctx.createLinearGradient(0, 0, 384, 384);
    bg.addColorStop(0, '#FF9A8B');
    bg.addColorStop(0.5, '#FF6A88');
    bg.addColorStop(1, '#4FACFE');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 384, 384);

    // Hair / silhouette
    ctx.fillStyle = '#2B1B54';
    ctx.beginPath();
    ctx.arc(192, 175, 115, 0, Math.PI * 2);
    ctx.fill();

    // Shoulders / jacket
    ctx.fillStyle = '#00C9A7';
    ctx.beginPath();
    ctx.ellipse(192, 360, 135, 95, 0, Math.PI, 0, false);
    ctx.fill();

    // Collar stripes
    ctx.strokeStyle = '#FFBE0B';
    ctx.lineWidth = 10;
    ctx.beginPath();
    ctx.moveTo(145, 275);
    ctx.lineTo(192, 345);
    ctx.lineTo(239, 275);
    ctx.stroke();

    // Face
    ctx.fillStyle = '#FFD1B3';
    ctx.beginPath();
    ctx.ellipse(192, 185, 72, 86, 0, 0, Math.PI * 2);
    ctx.fill();

    // Sunglasses
    ctx.fillStyle = '#FF007F';
    ctx.fillRect(128, 152, 56, 34);
    ctx.fillRect(200, 152, 56, 34);
    ctx.strokeStyle = '#1E1B4B';
    ctx.lineWidth = 6;
    ctx.strokeRect(128, 152, 56, 34);
    ctx.strokeRect(200, 152, 56, 34);
    ctx.beginPath();
    ctx.moveTo(184, 168);
    ctx.lineTo(200, 168);
    ctx.stroke();

    // Smile
    ctx.strokeStyle = '#D90429';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(192, 212, 24, 0.15 * Math.PI, 0.85 * Math.PI);
    ctx.stroke();

    return { dataUrl: canvas.toDataURL('image/png'), filename: 'sample-pop-portrait.png' };
  }

  if (preset === 'cybercar') {
    const bg = ctx.createLinearGradient(0, 0, 0, 384);
    bg.addColorStop(0, '#1A0B2E');
    bg.addColorStop(0.6, '#7928CA');
    bg.addColorStop(1, '#0F172A');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 384, 384);

    // Neon sun
    ctx.fillStyle = '#FFE600';
    ctx.beginPath();
    ctx.arc(192, 145, 76, 0, Math.PI * 2);
    ctx.fill();

    // Sports car body
    ctx.fillStyle = '#00F0FF';
    ctx.beginPath();
    ctx.moveTo(48, 265);
    ctx.lineTo(105, 200);
    ctx.lineTo(265, 200);
    ctx.lineTo(336, 255);
    ctx.lineTo(342, 295);
    ctx.lineTo(42, 295);
    ctx.closePath();
    ctx.fill();

    // Windshield
    ctx.fillStyle = '#FF007F';
    ctx.beginPath();
    ctx.moveTo(118, 208);
    ctx.lineTo(248, 208);
    ctx.lineTo(285, 248);
    ctx.lineTo(95, 248);
    ctx.closePath();
    ctx.fill();

    // Wheels
    ctx.fillStyle = '#1E293B';
    ctx.beginPath();
    ctx.arc(112, 295, 34, 0, Math.PI * 2);
    ctx.arc(276, 295, 34, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#FFE600';
    ctx.beginPath();
    ctx.arc(112, 295, 16, 0, Math.PI * 2);
    ctx.arc(276, 295, 16, 0, Math.PI * 2);
    ctx.fill();

    return { dataUrl: canvas.toDataURL('image/png'), filename: 'sample-cyber-car.png' };
  }

  // Default: 'landscape'
  const sky = ctx.createLinearGradient(0, 0, 0, 260);
  sky.addColorStop(0, '#3A0CA3');
  sky.addColorStop(0.5, '#F72585');
  sky.addColorStop(1, '#FFBE0B');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, 384, 384);

  // Sun
  ctx.fillStyle = '#FFF3B0';
  ctx.beginPath();
  ctx.arc(250, 120, 58, 0, Math.PI * 2);
  ctx.fill();

  // Mountains
  ctx.fillStyle = '#4361EE';
  ctx.beginPath();
  ctx.moveTo(0, 285);
  ctx.lineTo(115, 130);
  ctx.lineTo(235, 285);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = '#4CC9F0';
  ctx.beginPath();
  ctx.moveTo(140, 285);
  ctx.lineTo(265, 145);
  ctx.lineTo(384, 285);
  ctx.closePath();
  ctx.fill();

  // Foreground lake & trees
  ctx.fillStyle = '#06D6A0';
  ctx.fillRect(0, 275, 384, 109);

  ctx.fillStyle = '#118AB2';
  ctx.beginPath();
  ctx.ellipse(192, 335, 160, 32, 0, 0, Math.PI * 2);
  ctx.fill();

  return { dataUrl: canvas.toDataURL('image/png'), filename: 'sample-mountain-photo.png' };
}
