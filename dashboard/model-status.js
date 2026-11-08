/* Render actual model state from the backend; never present demo scores as ML output. */
window.MindCartModelStatus = {
  render(prefix, model, mode) {
    const title = document.getElementById(prefix + 'Status');
    const detail = document.getElementById(prefix + 'Detail');
    if (!title || !detail) return;
    if (mode === 'demo') {
      title.textContent = 'Synthetic demo · ML model not running';
      detail.textContent = 'The demo uses simulated scores.';
      return;
    }
    if (mode === 'disconnected') {
      title.textContent = 'Connect the EEG stream';
      detail.textContent = 'Model results appear here when available.';
      return;
    }
    if (!model) {
      title.textContent = 'Waiting for model data';
      detail.textContent = 'Start the project stream and a marked product trial.';
      return;
    }
    const item = model.item_id ? 'Item ' + model.item_id + ' · ' : '';
    const states = {
      waiting: 'Waiting for a product trial',
      collecting: 'Collecting EEG (' + (model.usable_windows || 0) + '/3 windows)',
      ready: 'Model output ready',
      insufficient_eeg: 'Not enough EEG for a model result',
      signal_interrupted: 'EEG signal interrupted',
      marker_delayed: 'Product timing unavailable'
    };
    title.textContent = item + (states[model.status] || 'Waiting for model data');
    const messages = [];
    if (model.status === 'insufficient_eeg') messages.push('The current model needs 15 seconds of usable item EEG.');
    if (model.video_liking != null) messages.push('Video-liking model: ' + (model.video_liking ? 'higher liking' : 'lower liking') + '.');
    if (model.impulse != null) messages.push('Impulse model: ' + (model.impulse ? 'impulse-related pattern' : 'non-impulse pattern') + '.');
    if (model.excitement != null) messages.push('Clothing excitement model: ' + (model.excitement ? 'higher' : 'lower') + '.');
    if (Array.isArray(model.available_models) && !model.available_models.includes('impulse')) messages.push('Clothing impulse model not trained yet.');
    if (model.rated_impulse != null) messages.push('Want/need answers: ' + (model.rated_impulse ? 'high want and low need' : 'other rating pattern') + '.');
    detail.textContent = messages.join(' ') || 'Waiting for a model result.';
  }
};
