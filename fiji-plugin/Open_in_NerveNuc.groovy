// Fiji script plugin: launch the same NerveNuc desktop analysis engine.
// Install this file under Fiji.app/scripts/Plugins/NerveNuc/.
import ij.IJ
import ij.Prefs
import ij.gui.GenericDialog
import ij.io.FileSaver

imp = IJ.getImage()
dialog = new GenericDialog('Open in NerveNuc')
dialog.addStringField('Python executable', Prefs.get('nervenuc.python', 'python'), 55)
dialog.addMessage('Use the Python environment where NerveNuc and its GUI dependencies are installed.')
dialog.showDialog()
if (!dialog.wasCanceled()) {
    executable = dialog.getNextString().trim()
    if (executable.length() == 0) {
        IJ.error('Choose the Python executable for NerveNuc.')
    } else {
        scratch = File.createTempFile('nervenuc-fiji-', '.tif')
        saver = new FileSaver(imp)
        saved = imp.getStackSize() > 1 ? saver.saveAsTiffStack(scratch.absolutePath) : saver.saveAsTiff(scratch.absolutePath)
        if (!saved) {
            IJ.error('Could not create a TIFF copy of the active image.')
        } else {
            try {
                log = new File(scratch.parentFile, scratch.name + '.log')
                builder = new ProcessBuilder([executable, '-m', 'nervenuc.gui.app', scratch.absolutePath] as String[])
                builder.redirectErrorStream(true)
                builder.redirectOutput(log)
                process = builder.start()
                Prefs.set('nervenuc.python', executable)
                IJ.showStatus('NerveNuc started. Original Fiji image remains unchanged.')
                IJ.log('NerveNuc input copy: ' + scratch.absolutePath)
                IJ.log('Launch log: ' + log.absolutePath)
            } catch (Exception error) {
                scratch.delete()
                IJ.error('Could not start NerveNuc: ' + error.message)
            }
        }
    }
}
