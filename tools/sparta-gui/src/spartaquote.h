// -*- c++ -*- /////////////////////////////////////////////////////////////////////////
// SPARTA-GUI - A Graphical Tool to Learn and Explore the SPARTA DSMC Simulation Software
//
// This software is distributed under the GNU General Public License version 2 or later.
////////////////////////////////////////////////////////////////////////////////////////

#ifndef SPARTAQUOTE_H
#define SPARTAQUOTE_H

#include <QString>

/**
 * @brief Quote a file path for use as one word of a SPARTA command
 *
 * SPARTA splits a command on whitespace, so "read_surf /Users/me/My Runs/x.surf"
 * is three arguments. A path with whitespace or a quote character is wrapped in
 * the quote style SPARTA's parser accepts that the path does not itself contain
 * (double, then single, then triple quotes); any other path is returned as is,
 * so commands shown to the user stay as plain as they were.
 */
inline QString quoteSpartaArg(const QString &arg)
{
    bool needs = arg.isEmpty();
    for (const QChar c : arg)
        if (c.isSpace() || c == '"' || c == '\'') needs = true;
    if (!needs) return arg;
    if (!arg.contains('"')) return '"' + arg + '"';
    if (!arg.contains('\'')) return '\'' + arg + '\'';
    return "\"\"\"" + arg + "\"\"\"";
}

#endif
